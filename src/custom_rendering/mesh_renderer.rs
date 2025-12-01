use alloc::sync::Arc;
use core::num::NonZeroU64;
use std::sync::Mutex;

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
				array_stride: core::mem::size_of::<egui::epaint::Vertex>() as wgpu::BufferAddress,
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

	wgpu_render_state.renderer.write().callback_resources.insert(MeshRenderResources {
		pipeline,
		bind_group,
		uniform_buffer,
	});

	Some(())
}

struct MeshRenderResources {
	pipeline:       wgpu::RenderPipeline,
	bind_group:     wgpu::BindGroup,
	uniform_buffer: wgpu::Buffer,
}

pub struct MeshCallback {
	mesh:          Arc<Mesh>,
	frame:         Rect,
	vertex_buffer: Mutex<Option<wgpu::Buffer>>,
	index_buffer:  Mutex<Option<wgpu::Buffer>>,
	index_count:   u32,
}

impl MeshCallback {
	pub fn new(mesh: Arc<Mesh>, frame: Rect) -> Self {
		Self {
			vertex_buffer: Mutex::new(None),
			index_buffer: Mutex::new(None),
			frame,
			index_count: mesh.indices.len() as u32,
			mesh,
		}
	}
}

impl egui_wgpu::CallbackTrait for MeshCallback {
	fn prepare(
		&self, device: &wgpu::Device, queue: &wgpu::Queue, screen_descriptor: &egui_wgpu::ScreenDescriptor,
		_egui_encoder: &mut wgpu::CommandEncoder, resources: &mut egui_wgpu::CallbackResources,
	) -> Vec<wgpu::CommandBuffer> {
		let mesh_resources: &MeshRenderResources = resources.get().unwrap();

		let uniforms = [
			screen_descriptor.size_in_pixels[0] as f32,
			screen_descriptor.size_in_pixels[1] as f32,
			self.frame.min.x,
			self.frame.min.y,
			self.frame.width(),
			self.frame.height(),
		];
		queue.write_buffer(&mesh_resources.uniform_buffer, 0, bytemuck::cast_slice(&uniforms));

		let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label:    Some("mesh_vertex_buffer"),
			contents: bytemuck::cast_slice(&self.mesh.vertices),
			usage:    wgpu::BufferUsages::VERTEX,
		});

		let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label:    Some("mesh_index_buffer"),
			contents: bytemuck::cast_slice(&self.mesh.indices),
			usage:    wgpu::BufferUsages::INDEX,
		});

		*self.index_buffer.lock().unwrap() = Some(index_buffer);
		*self.vertex_buffer.lock().unwrap() = Some(vertex_buffer);

		Vec::new()
	}

	fn paint(
		&self, _info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>,
		callback_resources: &egui_wgpu::CallbackResources,
	) {
		let mesh_resources: &MeshRenderResources = callback_resources.get().unwrap();

		let index_buffer = self.index_buffer.lock().unwrap();
		let vertex_buffer = self.vertex_buffer.lock().unwrap();
		let index_buffer = index_buffer.as_ref().unwrap();
		let vertex_buffer = vertex_buffer.as_ref().unwrap();

		render_pass.set_pipeline(&mesh_resources.pipeline);
		render_pass.set_bind_group(0, &mesh_resources.bind_group, &[]);
		render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
		render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
		render_pass.draw_indexed(0..self.index_count, 0, 0..1);
	}
}
