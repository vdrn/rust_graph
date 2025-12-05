use alloc::sync::Arc;
use send_wrapper::SendWrapper;
use core::cell::RefCell;
use core::num::NonZeroU64;

use eframe::egui::{self, Mesh, Rect};
use eframe::egui_wgpu::wgpu::util::DeviceExt as _;
use eframe::egui_wgpu::{self, wgpu};

pub fn init_mesh_renderer(cc: &eframe::CreationContext) -> Option<()> {
	let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
	let device = &wgpu_render_state.device;

	let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
		label:  Some("mesh_renderer"),
		source: wgpu::ShaderSource::Wgsl(include_str!("./mesh_shader.wgsl").into()),
	});

	let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
		label:   Some("mesh_renderer"),
		entries: &[wgpu::BindGroupLayoutEntry {
			binding:    0,
			visibility: wgpu::ShaderStages::VERTEX,
			ty:         wgpu::BindingType::Buffer {
				ty:                 wgpu::BufferBindingType::Uniform,
				has_dynamic_offset: false,
				min_binding_size:   NonZeroU64::new(24), // screen_size + rect(x, y, width, height)
			},
			count:      None,
		}],
	});

	let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		label:                Some("mesh_renderer"),
		bind_group_layouts:   &[&bind_group_layout],
		push_constant_ranges: &[],
	});

	let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
		label:         Some("mesh_renderer"),
		layout:        Some(&pipeline_layout),
		vertex:        wgpu::VertexState {
			module:              &shader,
			entry_point:         None,
			buffers:             &[wgpu::VertexBufferLayout {
				array_stride: size_of::<egui::epaint::Vertex>() as wgpu::BufferAddress,
				step_mode:    wgpu::VertexStepMode::Vertex,
				attributes:   &[
					// pos
					wgpu::VertexAttribute {
						offset:          0,
						shader_location: 0,
						format:          wgpu::VertexFormat::Float32x2,
					},
					// uv
					wgpu::VertexAttribute {
						offset:          8,
						shader_location: 1,
						format:          wgpu::VertexFormat::Float32x2,
					},
					// color
					wgpu::VertexAttribute {
						offset:          16,
						shader_location: 2,
						format:          wgpu::VertexFormat::Uint32,
					},
				],
			}],
			compilation_options: wgpu::PipelineCompilationOptions::default(),
		},
		fragment:      Some(wgpu::FragmentState {
			module:              &shader,
			entry_point:         Some("fs_main"),
			targets:             &[Some(wgpu::ColorTargetState {
				format:     wgpu_render_state.target_format,
				blend:      Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
				write_mask: wgpu::ColorWrites::ALL,
			})],
			compilation_options: wgpu::PipelineCompilationOptions::default(),
		}),
		primitive:     wgpu::PrimitiveState {
			topology: wgpu::PrimitiveTopology::TriangleList,
			..Default::default()
		},
		depth_stencil: None,
		multisample:   wgpu::MultisampleState::default(),
		multiview:     None,
		cache:         None,
	});

	let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label:    Some("mesh_renderer_uniform"),
		contents: bytemuck::cast_slice(&[0.0_f32; 6]),
		usage:    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
	});

	let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		label:   Some("mesh_renderer"),
		layout:  &bind_group_layout,
		entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
	});

	wgpu_render_state.renderer.write().callback_resources.insert(SendWrapper::new(MeshRenderResources {
		pipeline,
		bind_group,
		uniform_buffer,
		state: RefCell::new(BufferState {
			vertex_buffer:   None,
			index_buffer:    None,
			vertex_capacity: 0,
			index_capacity:  0,
			vertex_count:    0,
			index_count:     0,
      index_scratch:   Vec::with_capacity(1024),
		}),
	}));

	Some(())
}

struct BufferState {
	vertex_buffer:   Option<wgpu::Buffer>,
	vertex_capacity: usize,
	vertex_count:    usize,

	index_buffer:    Option<wgpu::Buffer>,
	index_capacity:  usize,
	index_count:     usize,

  /// used for offseting current mesh indices
  index_scratch:   Vec<u32>,
}

struct MeshRenderResources {
	pipeline:       wgpu::RenderPipeline,
	bind_group:     wgpu::BindGroup,
	uniform_buffer: wgpu::Buffer,
	state:          RefCell<BufferState>,
}

pub struct MeshCallback {
	mesh:  Arc<Mesh>,
	frame: Rect,
}

impl MeshCallback {
	pub fn new(mesh: Arc<Mesh>, frame: Rect) -> Self { Self { mesh, frame } }
}

impl egui_wgpu::CallbackTrait for MeshCallback {
	fn prepare(
		&self, device: &wgpu::Device, queue: &wgpu::Queue, screen_descriptor: &egui_wgpu::ScreenDescriptor,
		egui_encoder: &mut wgpu::CommandEncoder, resources: &mut egui_wgpu::CallbackResources,
	) -> Vec<wgpu::CommandBuffer> {
		let mesh_resources: &mut SendWrapper<MeshRenderResources> = resources.get_mut().unwrap();
    let mesh_resources: &mut MeshRenderResources = mesh_resources;
    let state = &mut mesh_resources.state.borrow_mut();


		if state.index_count == 0 {
			// uniform is always be the same within one frame, so we only need to update it the first time.
			let uniforms = [
				screen_descriptor.size_in_pixels[0] as f32,
				screen_descriptor.size_in_pixels[1] as f32,
				self.frame.min.x,
				self.frame.min.y,
				self.frame.width(),
				self.frame.height(),
			];
			queue.write_buffer(&mesh_resources.uniform_buffer, 0, bytemuck::cast_slice(&uniforms));
		}

		let new_vertex_count = state.vertex_count + self.mesh.vertices.len();
		let new_index_count = state.index_count + self.mesh.indices.len();

		// grow vertex buffer
		if new_vertex_count > state.vertex_capacity {
			let new_capacity = new_vertex_count.max(state.vertex_capacity * 2);
			let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
				label:              Some("mesh_vertex_buffer"),
				size:               (new_capacity * size_of::<egui::epaint::Vertex>()) as u64,
				usage:              wgpu::BufferUsages::VERTEX
					| wgpu::BufferUsages::COPY_DST
					| wgpu::BufferUsages::COPY_SRC,
				mapped_at_creation: false,
			});

			// copy old data
			if let Some(old_buffer) = &state.vertex_buffer {
				let copy_size = (state.vertex_count * size_of::<egui::epaint::Vertex>()) as u64;
				if copy_size > 0 {
					egui_encoder.copy_buffer_to_buffer(old_buffer, 0, &new_buffer, 0, copy_size);
				}
			}

			state.vertex_buffer = Some(new_buffer);
			state.vertex_capacity = new_capacity;
		}

		// grow index buffer
		if new_index_count > state.index_capacity {
			let new_capacity = new_index_count.max(state.index_capacity * 2);
			let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
				label:              Some("mesh_index_buffer"),
				size:               (new_capacity * size_of::<u32>()) as u64,
				usage:              wgpu::BufferUsages::INDEX
					| wgpu::BufferUsages::COPY_DST
					| wgpu::BufferUsages::COPY_SRC,
				mapped_at_creation: false,
			});

			// copy old data
			if let Some(old_buffer) = &state.index_buffer {
				let copy_size = (state.index_count * size_of::<u32>()) as u64;
				if copy_size > 0 {
					egui_encoder.copy_buffer_to_buffer(old_buffer, 0, &new_buffer, 0, copy_size);
				}
			}

			state.index_buffer = Some(new_buffer);
			state.index_capacity = new_capacity;
		}

		// write vertices 
		let vertex_offset = (state.vertex_count * size_of::<egui::epaint::Vertex>()) as u64;
		queue.write_buffer(
			state.vertex_buffer.as_ref().unwrap(),
			vertex_offset,
			bytemuck::cast_slice(&self.mesh.vertices),
		);

		// we have to adjust indices by the current vertex offset
		let base_vertex = state.vertex_count as u32;
    state.index_scratch.clear();
    for idx in self.mesh.indices.iter() {
      state.index_scratch.push(*idx + base_vertex);
    }
		let index_offset = (state.index_count * size_of::<u32>()) as u64;
		queue.write_buffer(
			state.index_buffer.as_ref().unwrap(),
			index_offset,
			bytemuck::cast_slice(&state.index_scratch),
		);

		state.vertex_count = new_vertex_count;
		state.index_count = new_index_count;

		Vec::new()
	}

	fn paint(
		&self, _info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>,
		callback_resources: &egui_wgpu::CallbackResources,
	) {
		let mesh_resources: &SendWrapper<MeshRenderResources> = callback_resources.get().unwrap();
    let mesh_resources :&MeshRenderResources = mesh_resources;
		let state = &mut mesh_resources.state.borrow_mut();

		// we draw everything in the first pain callback
		if state.index_count == 0 {
			return;
		}

		let vertex_buffer = state.vertex_buffer.as_ref().unwrap();
		let index_buffer = state.index_buffer.as_ref().unwrap();

		render_pass.set_pipeline(&mesh_resources.pipeline);
		render_pass.set_bind_group(0, &mesh_resources.bind_group, &[]);
		render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
		render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
		render_pass.draw_indexed(0..(state.index_count as u32), 0, 0..1);

    // reset counts back to 0, so we only draw in the first callback
		state.vertex_count = 0;
		state.index_count = 0;
	}
}
