use core::num::NonZeroU64;

use eframe::egui;
use eframe::egui_wgpu::wgpu::util::DeviceExt as _;
use eframe::egui_wgpu::{self, wgpu};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default, PartialEq, Clone, Copy, Debug)]
pub enum FillRule {
	#[default]
	EvenOdd,
	NonZero,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriangleFanVertex {
	x: f32,
	y: f32,
}

impl TriangleFanVertex {
	pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
}

pub struct CustomRenderer {
	texture_pool:          Vec<TextureResource>,
	current_texture_index: usize,
	stencil_texture:       Option<StencilTexture>,
}

struct TextureResource {
	texture_id: egui::TextureId,
	texture:    wgpu::Texture,
	width:      u32,
	height:     u32,
}

struct StencilTexture {
	view:   wgpu::TextureView,
	width:  u32,
	height: u32,
}

impl CustomRenderer {
	pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
		let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
		let device = &wgpu_render_state.device;

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label:  Some("fan_shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("./fan_shader.wgsl").into()),
		});

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label:   Some("fan_bind_group_layout"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding:    0,
				visibility: wgpu::ShaderStages::FRAGMENT,
				ty:         wgpu::BindingType::Buffer {
					ty:                 wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size:   NonZeroU64::new(16),
				},
				count:      None,
			}],
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label:                Some("fan_pipeline_layout"),
			bind_group_layouts:   &[&bind_group_layout],
			push_constant_ranges: &[],
		});
		let mut stencil_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
			label:         Some("stencil_pipeline"),
			layout:        Some(&pipeline_layout),
			vertex:        wgpu::VertexState {
				module:              &shader,
				entry_point:         Some("vs_main"),
				buffers:             &[wgpu::VertexBufferLayout {
					array_stride: size_of::<TriangleFanVertex>() as wgpu::BufferAddress,
					step_mode:    wgpu::VertexStepMode::Vertex,
					attributes:   &[wgpu::VertexAttribute {
						format:          wgpu::VertexFormat::Float32x2,
						offset:          0,
						shader_location: 0,
					}],
				}],
				compilation_options: wgpu::PipelineCompilationOptions::default(),
			},
			fragment:      None, // No color output in stencil pass
			primitive:     wgpu::PrimitiveState::default(),
			depth_stencil: Some(wgpu::DepthStencilState {
				format:              wgpu::TextureFormat::Stencil8,
				depth_write_enabled: false,
				depth_compare:       wgpu::CompareFunction::Always,
				stencil:             wgpu::StencilState {
					front:      wgpu::StencilFaceState {
						compare:       wgpu::CompareFunction::Always,
						fail_op:       wgpu::StencilOperation::Keep,
						depth_fail_op: wgpu::StencilOperation::Keep,
						pass_op:       wgpu::StencilOperation::Invert,
					},
					back:       wgpu::StencilFaceState {
						compare:       wgpu::CompareFunction::Always,
						fail_op:       wgpu::StencilOperation::Keep,
						depth_fail_op: wgpu::StencilOperation::Keep,
						pass_op:       wgpu::StencilOperation::Invert,
					},
					read_mask:  0xff,
					write_mask: 0xff,
				},
				bias:                wgpu::DepthBiasState::default(),
			}),
			multisample:   wgpu::MultisampleState::default(),
			multiview:     None,
			cache:         None,
		};
		let even_odd_stencil_pipeline = device.create_render_pipeline(&stencil_pipeline_descriptor);

		stencil_pipeline_descriptor.depth_stencil = Some(wgpu::DepthStencilState {
			format:              wgpu::TextureFormat::Stencil8,
			depth_write_enabled: false,
			depth_compare:       wgpu::CompareFunction::Always,
			stencil:             wgpu::StencilState {
				front:      wgpu::StencilFaceState {
					compare:       wgpu::CompareFunction::Always,
					fail_op:       wgpu::StencilOperation::Keep,
					depth_fail_op: wgpu::StencilOperation::Keep,
					pass_op:       wgpu::StencilOperation::IncrementWrap,
				},
				back:       wgpu::StencilFaceState {
					compare:       wgpu::CompareFunction::Always,
					fail_op:       wgpu::StencilOperation::Keep,
					depth_fail_op: wgpu::StencilOperation::Keep,
					pass_op:       wgpu::StencilOperation::DecrementWrap,
				},
				read_mask:  0xff,
				write_mask: 0xff,
			},
			bias:                wgpu::DepthBiasState::default(),
		});
		let non_zero_stencil_pipeline = device.create_render_pipeline(&stencil_pipeline_descriptor);

		let mut color_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
			label:         Some("color_pipeline"),
			layout:        Some(&pipeline_layout),
			vertex:        wgpu::VertexState {
				module:              &shader,
				entry_point:         Some("vs_quad"),
				buffers:             &[],
				compilation_options: wgpu::PipelineCompilationOptions::default(),
			},
			fragment:      Some(wgpu::FragmentState {
				module:              &shader,
				entry_point:         Some("fs_quad"),
				targets:             &[Some(wgpu::ColorTargetState {
					format:     wgpu::TextureFormat::Rgba8UnormSrgb,
					blend:      Some(wgpu::BlendState::REPLACE),
					write_mask: wgpu::ColorWrites::ALL,
				})],
				compilation_options: wgpu::PipelineCompilationOptions::default(),
			}),
			primitive:     wgpu::PrimitiveState::default(),
			depth_stencil: Some(wgpu::DepthStencilState {
				format:              wgpu::TextureFormat::Stencil8,
				depth_write_enabled: false,
				depth_compare:       wgpu::CompareFunction::Always,
				stencil:             wgpu::StencilState {
					front:      wgpu::StencilFaceState {
						compare:       wgpu::CompareFunction::NotEqual,
						fail_op:       wgpu::StencilOperation::Keep,
						depth_fail_op: wgpu::StencilOperation::Keep,
						pass_op:       wgpu::StencilOperation::Keep,
					},
					back:       wgpu::StencilFaceState {
						compare:       wgpu::CompareFunction::NotEqual,
						fail_op:       wgpu::StencilOperation::Keep,
						depth_fail_op: wgpu::StencilOperation::Keep,
						pass_op:       wgpu::StencilOperation::Keep,
					},
					read_mask:  0xff,
					write_mask: 0x00,
				},
				bias:                wgpu::DepthBiasState::default(),
			}),
			multisample:   wgpu::MultisampleState::default(),
			multiview:     None,
			cache:         None,
		};
		let color_pipeline = device.create_render_pipeline(&color_pipeline_descriptor);

		let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label:    Some("uniform_buffer"),
			contents: bytemuck::cast_slice(&[0.0_f32; 4]),
			usage:    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
		});

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label:   Some("bind_group"),
			layout:  &bind_group_layout,
			entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
		});

		let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label:              Some("vertex_buffer"),
			size:               1024,
			usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label:              Some("index_buffer"),
			size:               1024,
			usage:              wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		wgpu_render_state.renderer.write().callback_resources.insert(FanRenderResources {
			even_odd_stencil_pipeline,
			non_zero_stencil_pipeline,
			color_pipeline,
			bind_group,
			uniform_buffer,
			vertex_buffer,
			index_buffer,
			vertex_buffer_capacity: 1024,
			index_buffer_capacity: 1024,
			indices: Vec::with_capacity(1024 * 3),
		});

		Some(Self { texture_pool: Vec::new(), current_texture_index: 0, stencil_texture: None })
	}

	pub fn reset_textures(&mut self) { self.current_texture_index = 0; }

	pub fn paint_curve_fill(
		&mut self, render_state: &egui_wgpu::RenderState, vertices: &[TriangleFanVertex],
		color: egui::Color32, fill_rule: FillRule, size_x: f32, size_y: f32,
	) -> egui::TextureId {
		let width = size_x as u32;
		let height = size_y as u32;

		assert!(vertices.len() > 2);

		self.ensure_stencil_buffer(render_state, width, height);

		let texture_resource = Self::get_or_create_texture(
			self.current_texture_index, &mut self.texture_pool, render_state, width, height,
		);

		Self::render_to_texture(
			render_state,
			texture_resource,
			self.stencil_texture.as_ref().unwrap(),
			vertices,
			color,
			fill_rule,
		);

		self.current_texture_index += 1;

		texture_resource.texture_id
	}

	fn get_or_create_texture<'a>(
		current_texture_index: usize, texture_pool: &'a mut Vec<TextureResource>,
		render_state: &egui_wgpu::RenderState, width: u32, height: u32,
	) -> &'a mut TextureResource {
		if current_texture_index < texture_pool.len() {
			let resource = &mut texture_pool[current_texture_index];
			if resource.width != width || resource.height != height {
				*resource = Self::create_texture_resource(render_state, width, height);
			}
			resource
		} else {
			texture_pool.push(Self::create_texture_resource(render_state, width, height));
			texture_pool.last_mut().unwrap()
		}
	}

	/// ensure stencil buffer matches current size
	fn ensure_stencil_buffer(&mut self, render_state: &egui_wgpu::RenderState, width: u32, height: u32) {
		let needs_recreate =
			self.stencil_texture.as_ref().is_none_or(|st| st.width != width || st.height != height);

		if needs_recreate {
			let device = &render_state.device;
			let texture = device.create_texture(&wgpu::TextureDescriptor {
				label:           Some("stencil_texture"),
				size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
				mip_level_count: 1,
				sample_count:    1,
				dimension:       wgpu::TextureDimension::D2,
				format:          wgpu::TextureFormat::Stencil8,
				usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
				view_formats:    &[],
			});

			let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

			self.stencil_texture = Some(StencilTexture { view, width, height });
		}
	}

	fn create_texture_resource(
		render_state: &egui_wgpu::RenderState, width: u32, height: u32,
	) -> TextureResource {
		let device = &render_state.device;

		let texture = device.create_texture(&wgpu::TextureDescriptor {
			label:           Some("custom_render_texture"),
			size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count:    1,
			dimension:       wgpu::TextureDimension::D2,
			format:          wgpu::TextureFormat::Rgba8UnormSrgb,
			usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
			view_formats:    &[],
		});

		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

		let texture_id =
			render_state.renderer.write().register_native_texture(device, &view, wgpu::FilterMode::Linear);

		TextureResource { texture_id, texture, width, height }
	}

	fn render_to_texture(
		render_state: &egui_wgpu::RenderState, texture_resource: &TextureResource,
		stencil_texture: &StencilTexture, vertices: &[TriangleFanVertex], color: egui::Color32,
		fill_rule: FillRule,
	) {
		let device = &render_state.device;
		let queue = &render_state.queue;

		let color_view = texture_resource.texture.create_view(&wgpu::TextureViewDescriptor::default());

		let color_unif = [
			color.r() as f32 / 255.0,
			color.g() as f32 / 255.0,
			color.b() as f32 / 255.0,
			color.a() as f32 / 255.0,
		];

		let mut renderer = render_state.renderer.write();
		let resources: &mut FanRenderResources = renderer.callback_resources.get_mut().unwrap();

		queue.write_buffer(&resources.uniform_buffer, 0, bytemuck::cast_slice(&color_unif));

		// update vertex buffer
		let vertex_data = bytemuck::cast_slice(vertices);
		if vertex_data.len() > resources.vertex_buffer_capacity {
			// must resize vertex buffer
			let new_capacity = vertex_data.len().next_power_of_two();
			resources.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
				label:              Some("vertex_buffer"),
				size:               new_capacity as u64,
				usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
			resources.vertex_buffer_capacity = new_capacity;
		}
		queue.write_buffer(&resources.vertex_buffer, 0, vertex_data);

		let num_triangles = vertices.len() - 2;
		let num_indices = num_triangles * 3;
		let prev_num_indices = resources.indices.len();
		if prev_num_indices < num_indices {
			#[allow(clippy::integer_division)]
			let current_triangles = resources.indices.len() / 3;
			for i in current_triangles..num_triangles {
				resources.indices.push(0u32);
				resources.indices.push((i + 1) as u32);
				resources.indices.push((i + 2) as u32);
			}
		}
		assert!(resources.indices.len() >= num_indices);

		let index_data = bytemuck::cast_slice(&resources.indices[0..num_indices]);
		if index_data.len() > resources.index_buffer_capacity {
			// must resize index buffer
			let new_capacity = index_data.len().next_power_of_two();
			resources.index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
				label:              Some("index_buffer"),
				size:               new_capacity as u64,
				usage:              wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
			resources.index_buffer_capacity = new_capacity;
		}

		if prev_num_indices < num_indices {
			// must update the index buffer with new indices
			queue.write_buffer(&resources.index_buffer, 0, index_data);
		}

		drop(renderer);

		let mut encoder = device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fan_render_encoder") });

		// PASS 1: draw stencil buffer
		{
			let mut stencil_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label:                    Some("stencil_pass"),
				color_attachments:        &[],
				depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
					view:        &stencil_texture.view,
					depth_ops:   None,
					stencil_ops: Some(wgpu::Operations {
						load:  wgpu::LoadOp::Clear(0),
						store: wgpu::StoreOp::Store,
					}),
				}),
				timestamp_writes:         None,
				occlusion_query_set:      None,
			});

			let renderer = render_state.renderer.read();
			let resources: &FanRenderResources = renderer.callback_resources.get().unwrap();

			match fill_rule {
				FillRule::EvenOdd => {
					stencil_pass.set_pipeline(&resources.even_odd_stencil_pipeline);
				},
				FillRule::NonZero => {
					stencil_pass.set_pipeline(&resources.non_zero_stencil_pipeline);
				},
			}
			stencil_pass.set_bind_group(0, &resources.bind_group, &[]);
			stencil_pass.set_vertex_buffer(0, resources.vertex_buffer.slice(..));
			stencil_pass.set_index_buffer(resources.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
			stencil_pass.set_stencil_reference(0);
			stencil_pass.draw_indexed(0..(num_indices as u32), 0, 0..1);
		}

		// PASS 2: draw quad with stencil test
		{
			let mut color_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label:                    Some("color_pass"),
				color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
					view:           &color_view,
					resolve_target: None,
					ops:            wgpu::Operations {
						load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
						store: wgpu::StoreOp::Store,
					},
					depth_slice:    None,
				})],
				depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
					view:        &stencil_texture.view,
					depth_ops:   None,
					stencil_ops: Some(wgpu::Operations {
						load:  wgpu::LoadOp::Load,
						store: wgpu::StoreOp::Discard,
					}),
				}),
				timestamp_writes:         None,
				occlusion_query_set:      None,
			});

			let renderer = render_state.renderer.read();
			let resources: &FanRenderResources = renderer.callback_resources.get().unwrap();

			color_pass.set_pipeline(&resources.color_pipeline);
			color_pass.set_bind_group(0, &resources.bind_group, &[]);
			color_pass.set_stencil_reference(0);
			color_pass.draw(0..6, 0..1);
		}

		queue.submit(Some(encoder.finish()));
	}
}

struct FanRenderResources {
	even_odd_stencil_pipeline: wgpu::RenderPipeline,
	non_zero_stencil_pipeline: wgpu::RenderPipeline,
	color_pipeline:            wgpu::RenderPipeline,
	bind_group:                wgpu::BindGroup,
	uniform_buffer:            wgpu::Buffer,
	vertex_buffer:             wgpu::Buffer,
	index_buffer:              wgpu::Buffer,
	vertex_buffer_capacity:    usize,
	index_buffer_capacity:     usize,
	indices:                   Vec<u32>,
}

#[cfg(target_arch = "wasm32")]
/// SAFETY:wgpu structs are not Send/Sync on wasm.
/// We need them to be to be able to put them into TypeMap.
/// We're not accessing them from other threads, so we're (hopefully) fine.
unsafe impl Sync for FanRenderResources {}
#[cfg(target_arch = "wasm32")]
/// SAFETY: same as above
unsafe impl Send for FanRenderResources {}
