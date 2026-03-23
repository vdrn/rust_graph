use alloc::sync::Arc;
use core::cell::RefCell;
use core::ptr;
use core::time::Duration;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::{Mutex, mpsc};
use std::time::Instant;

use eframe::egui::{self, Color32, Id, Mesh, Shape, Stroke};
use eframe::egui_wgpu;
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint, PlotPoints, Points};
use evalexpr::{EvalexprFloat, Stack};
use thread_local::ThreadLocal;

use crate::custom_rendering::fan_fill_renderer::{FillRule, TriangleFanVertex};
use crate::custom_rendering::mesh_renderer::MeshCallback;
use crate::drawing::line::DrawLine2;
use crate::drawing::{TextPlotItem, line};
use crate::entry::{Entry, EntryType, ProcessedColors};
use crate::marching_squares::MeshBuilder;
use crate::math::{closest_point_on_segment, dist_sq, intersect_segs};
use crate::thread_local_get;

pub struct ProcessedShapes {
	tesselator: line::Tessellator,

	pub lines:         Vec<ProcessedShape>,
	pub draw_points:   Vec<DrawPoint>,
	pub draw_polygons: Vec<DrawPolygonGroup>,
	pub draw_texts:    Vec<DrawText>,
	pub draw_meshes:   Vec<DrawMesh>,
}
#[derive(Clone)]
pub struct ProcessedShape {
	// pub shapes:      RefCell<Vec<Shape>>,
	pub shapes:      Arc<Mesh>,
	pub color:       Color32,
	pub bounds:      egui_plot::PlotBounds,
	pub base:        egui_plot::PlotItemBase,
	pub id:          Id,
	pub allow_hover: bool,
}
impl PlotItem for ProcessedShape {
	fn shapes(&self, _ui: &eframe::egui::Ui, transform: &egui_plot::PlotTransform, shapes: &mut Vec<Shape>) {
		// shapes.push(Shape::mesh(self.shapes.clone()));
		if !self.shapes.vertices.is_empty() {
			let rect = transform.frame();
			let callback = MeshCallback::new(self.shapes.clone(), *rect);
			shapes.push(Shape::Callback(egui_wgpu::Callback::new_paint_callback(*rect, callback)));
		}
	}

	fn initialize(&mut self, _x_range: core::ops::RangeInclusive<f64>) {}

	fn color(&self) -> Color32 { self.color }

	fn geometry(&self) -> PlotGeometry<'_> { PlotGeometry::None }

	fn bounds(&self) -> egui_plot::PlotBounds { self.bounds }

	fn base(&self) -> &egui_plot::PlotItemBase { &self.base }

	fn base_mut(&mut self) -> &mut egui_plot::PlotItemBase { &mut self.base }
	fn id(&self) -> Id { self.id }
	fn allow_hover(&self) -> bool { self.allow_hover }
}
impl ProcessedShapes {
	pub fn new() -> Self {
		Self {
			// tesselator:           eframe::epaint::Tessellator::new(
			// 	1.0,
			// 	TessellationOptions::default(),
			// 	[12, 12],
			// 	vec![],
			// ),
			tesselator:    line::Tessellator::new(1.0),
			lines:         Vec::with_capacity(16),
			draw_points:   Vec::with_capacity(16),
			draw_polygons: Vec::with_capacity(16),
			draw_texts:    Vec::with_capacity(16),
			draw_meshes:   Vec::with_capacity(16),
		}
	}
	pub fn process<T: EvalexprFloat>(
		&mut self, ui: &eframe::egui::Ui, entries: &mut [Entry<T>], plot_params: &crate::entry::PlotParams,
		selected_plot_line: Option<Id>,
	) {
		let ppp = ui.pixels_per_point();
		self.tesselator.set_pixels_per_point(ppp);

		self.lines.clear();
		self.draw_points.clear();
		self.draw_polygons.clear();
		self.draw_texts.clear();
		self.draw_meshes.clear();

		// let mut result = Vec::new();
		if let Some(transform) = plot_params.prev_plot_transform {
			let mut clone_draw_buffer = |draw_buffer: &mut DrawBuffer| {
				let mut bounds = PlotBounds::NOTHING;
				let mut name = None;
				let mut allow_hover = false;
				if !draw_buffer.lines.is_empty() {
					let mut mesh = Mesh::default();
					let mut id = Id::NULL;
					let mut color = Color32::TRANSPARENT;
					for line in draw_buffer.lines.iter_mut() {
						id = line.id;
						color = line.color;
						if name.is_none() {
							name = Some(line.name.clone());
						}
						allow_hover |= line.allow_hover;

						if Some(line.id) == selected_plot_line {
							let prev_width = line.width;
							line.width += 2.0;
							bounds.merge(&self.tesselator.tessalate_line(line, &transform, &mut mesh));
							line.width = prev_width;
						} else {
							bounds.merge(&self.tesselator.tessalate_line(line, &transform, &mut mesh));
							// bounds.merge(&line.tessalate(&transform, &mut mesh));
						}
						// for shape in self.shapes_temp.drain(..) {
						// 	self.tesselator.tessellate_shape(shape, &mut mesh);
						// }
					}
					let base = PlotItemBase::new(name.unwrap_or_default());
					self.lines.push(ProcessedShape {
						id,
						shapes: mesh.into(),
						allow_hover,
						color,
						bounds,
						base,
					});
				}
				for mesh in draw_buffer.meshes.iter() {
					self.draw_meshes.push(mesh.clone());
				}
				for polygon in draw_buffer.polygons.iter() {
					self.draw_polygons.push(polygon.clone());
				}
				for point in draw_buffer.points.iter() {
					self.draw_points.push(point.clone());
				}
				for text in draw_buffer.texts.iter() {
					self.draw_texts.push(text.clone());
				}
				// for mesh in draw_buffer.meshes.iter() {
				// draw_meshes.extend(draw_buffer.meshes.iter().cloned());
				// draw_polygons.extend(draw_buffer.polygons.iter().cloned());

				// draw_lines.extend(draw_buffer.lines.iter().cloned());
				// draw_points.extend(draw_buffer.points.iter().cloned());
				// draw_texts.extend(draw_buffer.texts.iter().cloned());
			};
			for entry in entries.iter_mut() {
				if let EntryType::Folder { entries } = &mut entry.ty {
					for entry in entries.iter_mut() {
						let draw_buffer = &mut entry.draw_buffer;
						clone_draw_buffer(draw_buffer);
					}
				} else {
					let draw_buffer = &mut entry.draw_buffer;
					clone_draw_buffer(draw_buffer);
				}
			}
		}
	}
}

#[repr(align(128))]
pub struct DrawBufferRC {
	pub inner: RefCell<DrawBuffer>,
}
impl Default for DrawBufferRC {
	fn default() -> Self { Self { inner: RefCell::new(DrawBuffer::default()) } }
}

pub struct DrawBuffer {
	pub lines:    Vec<DrawLine2>,
	pub points:   Vec<DrawPoint>,
	pub polygons: Vec<DrawPolygonGroup>,
	pub texts:    Vec<DrawText>,
	pub meshes:   Vec<DrawMesh>,
}
#[allow(clippy::non_send_fields_in_send_ty)]
/// SAFETY: Line/Points/Polygon are not Send/Sync because of `ExplicitGenerator` callbacks.
/// We dont use those so we're fine.
unsafe impl Send for DrawBuffer {}

impl Default for DrawBuffer {
	fn default() -> Self {
		Self {
			lines:    Vec::with_capacity(32),
			points:   Vec::with_capacity(32),
			polygons: Vec::with_capacity(4),
			texts:    Vec::with_capacity(4),
			meshes:   Vec::with_capacity(8),
		}
	}
}
impl DrawBuffer {
	pub fn empty() -> Self {
		Self {
			lines:    Vec::new(),
			points:   Vec::new(),
			polygons: Vec::new(),
			texts:    Vec::new(),
			meshes:   Vec::new(),
		}
	}
	pub fn clear(&mut self) {
		self.lines.clear();
		self.points.clear();
		self.polygons.clear();
		self.texts.clear();
		self.meshes.clear();
	}
}

pub struct ProcessDrawBuffersResult {
	pub closest_point_to_mouse_on_selected: Option<((f64, f64), f64)>,
	pub hovered_id:                         Option<Id>,
	// pub draw_lines:             Vec<DrawLine>,
	pub draw_points:                        Vec<DrawPoint>,
	// pub draw_polygons:          Vec<DrawPolygonGroup>,
	// pub draw_texts:             Vec<DrawText>,
	// pub draw_meshes:            Vec<DrawMesh>,
}
pub fn process_draw_buffers<T: EvalexprFloat>(
	entries: &[Entry<T>], selected_plot_line: Option<(Id, bool)>, mouse_pos_in_graph: Option<(f64, f64)>,
	plot_params: &crate::entry::PlotParams, draw_buffers: &mut ThreadLocal<DrawBufferRC>,
) -> ProcessDrawBuffersResult {
	let mut draw_points = vec![];

	let selected_entry = selected_plot_line.and_then(|(selected_fline_id, show_closest_point_to_mouse)| {
		entries.iter().find_map(|e| match &e.ty {
			EntryType::Folder { entries } => entries
				.iter()
				.find(|e| Id::new(e.id) == selected_fline_id)
				.map(|e| (e, show_closest_point_to_mouse)),
			_ => {
				if Id::new(e.id) == selected_fline_id {
					Some((e, show_closest_point_to_mouse))
				} else {
					None
				}
			},
		})
	});

	// let mut hovered_point: Option<Id> = None;
	let mut hovered_line: Option<(Id, f64)> = None;

	// let mut hover=|p1:PlotPoint,p2:PlotPoint|{
	//   if let Some(mouse_pos)= mouse_pos_in_graph{

	//   }

	// };

	let closest_point_to_mouse_for_selected: Mutex<Option<((f64, f64), f64)>> = Mutex::new(None);

	let mut process_entry = |entry: &Entry<T>| {
		let id = Id::new(entry.id);
		let mut closest_point_to_mouse: Option<((f64, f64), f64, Id)> = None;
		let mut hover = |prev: (f64, f64), cur: (f64, f64)| {
			if let Some(mouse_pos_in_graph) = mouse_pos_in_graph {
				let mouse_on_seg =
					closest_point_on_segment((cur.0, cur.1), (prev.0, prev.1), mouse_pos_in_graph);
				let dist_sq = dist_sq(mouse_on_seg, mouse_pos_in_graph);
				if let Some(cur_closest_point_to_mouse) = closest_point_to_mouse {
					if dist_sq < cur_closest_point_to_mouse.1 {
						closest_point_to_mouse = Some((mouse_on_seg, dist_sq, id));
					}
				} else {
					closest_point_to_mouse = Some((mouse_on_seg, dist_sq, id));
				}
			}
		};

		if let Some((selected_entry, show_closest_point_to_mouse)) = selected_entry
			&& ptr::eq(entry, selected_entry)
		{
			let draw_buffer = &selected_entry.draw_buffer;
			for line in draw_buffer.lines.iter() {
				let plot_points = line.points();
				let pt_radius = line.width + 2.5;

				// Find local optima
				let mut prev: [Option<(f64, f64)>; 2] = [None; 2];
				for (pi, (point, _)) in plot_points.iter().enumerate() {
					let cur = (point.x, point.y);
					fn less_then(a: f64, b: f64, e: f64) -> bool { b - a > e }
					fn greater_then(a: f64, b: f64, e: f64) -> bool { a - b > e }
					// Find closest point to mouse
					if let Some(prev_1) = prev[1] {
						if line.allow_hover {
							hover(prev_1, cur);
						}
						// if show_closest_point_to_mouse && let Some(mouse_pos_in_graph) =
						// mouse_pos_in_graph {
						// 	let mouse_on_seg = closest_point_on_segment(
						// 		(cur.0, cur.1),
						// 		(prev_1.0, prev_1.1),
						// 		mouse_pos_in_graph,
						// 	);
						// 	let dist_sq = dist_sq(mouse_on_seg, mouse_pos_in_graph);
						// 	if let Some(cur_closest_point_to_mouse) = closest_point_to_mouse {
						// 		if dist_sq < cur_closest_point_to_mouse.1 {
						// 			closest_point_to_mouse = Some((mouse_on_seg, dist_sq, id));
						// 		}
						// 	} else {
						// 		closest_point_to_mouse = Some((mouse_on_seg, dist_sq, id));
						// 	}
						// }

						if let Some(prev_0) = prev[0] {
							if prev_0.1.signum() != prev_1.1.signum() {
								// intersection with x axis
								let sum = prev_0.1.abs() + prev_1.1.abs();
								let x = if sum == 0.0 {
									(prev_0.0 + prev_1.0) * 0.5
								} else {
									let t = prev_0.1.abs() / sum;
									prev_0.0 + t * (prev_1.0 - prev_0.0)
								};
								draw_points.push(DrawPoint::new(
									line.sorting_index,
									pi as u32,
									PointInteraction {
										x,
										y: 0.0,
										radius: pt_radius,
										ty: PointInteractionType::Other(OtherPointType::IntersectionWithXAxis),
									},
									Points::new("", [x, 0.0]).color(Color32::GRAY).radius(pt_radius),
								));
							}

							if prev_0.0.signum() != prev_1.0.signum() {
								// intersection with y axis
								let sum = prev_0.0.abs() + prev_1.0.abs();
								let y = if sum == 0.0 {
									(prev_0.1 + prev_1.1) * 0.5
								} else {
									let t = prev_0.0.abs() / sum;
									prev_0.1 + t * (prev_1.1 - prev_0.1)
								};
								draw_points.push(DrawPoint::new(
									line.sorting_index,
									pi as u32,
									PointInteraction {
										x: 0.0,
										y,
										radius: pt_radius,
										ty: PointInteractionType::Other(OtherPointType::IntersectionWithYAxis),
									},
									Points::new("", [0.0, y]).color(Color32::GRAY).radius(pt_radius),
								));
							}

							if less_then(prev_0.1, prev_1.1, plot_params.eps)
								&& greater_then(prev_1.1, cur.1, plot_params.eps)
							{
								// local maximum
								draw_points.push(DrawPoint::new(
									line.sorting_index,
									pi as u32,
									PointInteraction {
										x:      prev_1.0,
										y:      prev_1.1,
										radius: pt_radius,
										ty:     PointInteractionType::Other(OtherPointType::Maxima),
									},
									Points::new("", [prev_1.0, prev_1.1])
										.color(Color32::GRAY)
										.radius(pt_radius),
								));
							}
							if greater_then(prev_0.1, prev_1.1, plot_params.eps)
								&& less_then(prev_1.1, cur.1, plot_params.eps)
							{
								// local minimum
								draw_points.push(DrawPoint::new(
									line.sorting_index,
									pi as u32,
									PointInteraction {
										x:      prev_1.0,
										y:      prev_1.1,
										radius: pt_radius,
										ty:     PointInteractionType::Other(OtherPointType::Minima),
									},
									Points::new("", [prev_1.0, prev_1.1])
										.color(Color32::GRAY)
										.radius(pt_radius),
								));
							}
						}
					}

					prev[0] = prev[1];
					prev[1] = Some(cur);
				}
			}
			if show_closest_point_to_mouse {
				if let Some((closest_point, dist_sq, _)) = closest_point_to_mouse {
					let mut final_closest_point_to_mouse = closest_point_to_mouse_for_selected.lock().unwrap();
					if let Some(cur_closest_point) = *final_closest_point_to_mouse {
						if dist_sq < cur_closest_point.1 {
							*final_closest_point_to_mouse = Some((closest_point, dist_sq));
						}
					} else {
						*final_closest_point_to_mouse = Some((closest_point, dist_sq));
					}
				}
			}
		} else {
			let draw_buffer = &entry.draw_buffer;
			for fline in draw_buffer.lines.iter() {
				// draw_lines.par_iter().for_each(|fline| {
				let plot_points = fline.points();
				let mut draw_buffer = thread_local_get(draw_buffers).inner.borrow_mut();
				if !fline.allow_hover {
					continue;
				}
				// find intersections
				let mut pi = 0;
				for plot_seg in plot_points.windows(2) {
					hover((plot_seg[0].0.x, plot_seg[0].0.y), (plot_seg[1].0.x, plot_seg[1].0.y));

					if let Some(selected_draw_buffer) = selected_entry.map(|(s, _)| &s.draw_buffer) {
						for selected_line in selected_draw_buffer.lines.iter() {
							let sel_points = selected_line.points();
							let pt_radius = selected_line.width + 2.5;
							for sel_seg in sel_points.windows(2) {
								if let Some(point) = intersect_segs(
									plot_seg[0].0, plot_seg[1].0, sel_seg[0].0, sel_seg[1].0, plot_params.eps,
								) {
									draw_buffer.points.push(DrawPoint::new(
										fline.sorting_index,
										pi as u32,
										PointInteraction {
											x:      point.x,
											y:      point.y,
											radius: pt_radius,
											ty:     PointInteractionType::Other(OtherPointType::Intersection),
										},
										Points::new("", [point.x, point.y])
											.color(Color32::GRAY)
											.radius(pt_radius),
									));
									pi += 1;
								}
							}
						}
					}
				}
			}
		}

		if let Some((_closest_point, dist_sq, id)) = closest_point_to_mouse {
			if let Some((_, cur_dist_sq)) = hovered_line {
				if dist_sq < cur_dist_sq {
					hovered_line = Some((id, dist_sq));
				}
			} else {
				hovered_line = Some((id, dist_sq));
			}
		}
	};
	for etry in entries.iter() {
		match &etry.ty {
			EntryType::Folder { entries } => {
				for entry in entries.iter() {
					process_entry(entry);
				}
			},
			_ => {
				process_entry(etry);
			},
		}
	}
	for draw_buffer in draw_buffers.iter_mut() {
		for draw_point in draw_buffer.inner.get_mut().points.drain(..) {
			draw_points.push(draw_point);
		}
	}

	let closest_point_to_mouse_for_selected = closest_point_to_mouse_for_selected.into_inner().unwrap();
	if let Some(closest_point_to_mouse_for_selected) = closest_point_to_mouse_for_selected {
		draw_points.push(DrawPoint::new(
			0,
			0,
			PointInteraction {
				x:      closest_point_to_mouse_for_selected.0.0,
				y:      closest_point_to_mouse_for_selected.0.1,
				radius: 5.0,
				ty:     PointInteractionType::Other(OtherPointType::Point),
			},
			Points::new(
				"",
				[closest_point_to_mouse_for_selected.0.0, closest_point_to_mouse_for_selected.0.1],
			)
			.color(Color32::GRAY)
			.radius(5.0),
		));
	}

	draw_points.sort_unstable_by_key(|draw_point| draw_point.sorting_index);

	let hovered_id = if let (Some(transform), Some((hovered_line_id, hovered_dist_sq))) =
		(&plot_params.prev_plot_transform, hovered_line)
	{
		let ui_dist = (transform.dpos_dvalue()[0].powi(2) + transform.dpos_dvalue()[1].powi(2)).sqrt();
		// println!("hovered_line: {hovered_line:?} {ui_dist:?}");
		if hovered_dist_sq.sqrt() * ui_dist < 4.0 { Some(hovered_line_id) } else { None }
	} else {
		None
	};

	ProcessDrawBuffersResult {
		closest_point_to_mouse_on_selected: closest_point_to_mouse_for_selected,
		draw_points,
		hovered_id,
	}
}

#[derive(Clone)]
pub struct DrawMesh {
	pub ty: DrawMeshType,
}
#[derive(Default, Clone)]
pub struct FillMesh {
	pub(crate) indices:             Vec<u32>,
	pub(crate) vertices:            Vec<TriangleFanVertex>,
	pub(crate) current_root_vertex: usize,
	pub(crate) texture_id:          Option<egui::TextureId>,
	pub(crate) color:               Color32,
	pub(crate) fill_rule:           FillRule,
}
impl FillMesh {
	pub fn new(color: Color32, fill_rule: FillRule) -> Self {
		Self {
			fill_rule,
			vertices: Vec::new(),
			indices: Vec::new(),
			texture_id: None,
			color,
			current_root_vertex: 0,
		}
	}
	pub fn add_vertex(&mut self, x: f32, y: f32) {
		self.vertices.push(TriangleFanVertex::new(x, y));
		if self.current_root_vertex + 2 < self.vertices.len() {
			self.indices.push(self.current_root_vertex as u32);
			self.indices.push(self.vertices.len() as u32 - 2);
			self.indices.push(self.vertices.len() as u32 - 1);
		}
	}
	pub fn reset_root_vertex(&mut self) { self.current_root_vertex = self.vertices.len(); }
}

#[derive(Clone)]
pub enum DrawMeshType {
	EguiPlotMesh(EguiPlotMesh),
	FillMesh(FillMesh),
}
#[derive(Clone)]
pub struct EguiPlotMesh {
	pub bounds:         egui_plot::PlotBounds,
	pub plot_item_base: egui_plot::PlotItemBase,
	pub mesh:           RefCell<MeshBuilder>,
	pub color:          Color32,
}
impl PlotItem for EguiPlotMesh {
	fn shapes(&self, _ui: &eframe::egui::Ui, transform: &egui_plot::PlotTransform, shapes: &mut Vec<Shape>) {
		for vertex in self.mesh.borrow_mut().vertices.iter_mut() {
			vertex.pos =
				transform.position_from_point(&PlotPoint::new(vertex.pos.x as f64, vertex.pos.y as f64));
		}
		let mesh = core::mem::take(&mut *self.mesh.borrow_mut());
		let mesh = Mesh { vertices: mesh.vertices, indices: mesh.indices, ..Default::default() };

		shapes.push(Shape::mesh(mesh));
	}

	fn initialize(&mut self, _x_range: core::ops::RangeInclusive<f64>) {}

	fn color(&self) -> Color32 { self.color }

	fn geometry(&self) -> PlotGeometry<'_> { PlotGeometry::None }

	fn bounds(&self) -> egui_plot::PlotBounds { self.bounds }

	fn base(&self) -> &egui_plot::PlotItemBase { &self.plot_item_base }

	fn base_mut(&mut self) -> &mut egui_plot::PlotItemBase { &mut self.plot_item_base }
}
#[derive(Clone, Debug)]
pub struct PointInteraction {
	pub ty:     PointInteractionType,
	pub x:      f64,
	pub y:      f64,
	pub radius: f32,
}
impl PointInteraction {
	pub fn name(&self) -> &'static str {
		match self.ty {
			PointInteractionType::Draggable { .. } | PointInteractionType::Other(OtherPointType::Point) => {
				"Point"
			},
			PointInteractionType::Other(OtherPointType::IntersectionWithXAxis) => "Intersection with x axis",
			PointInteractionType::Other(OtherPointType::IntersectionWithYAxis) => "Intersection with y axis",
			PointInteractionType::Other(OtherPointType::Intersection) => "Intersection",
			PointInteractionType::Other(OtherPointType::Minima) => "Minima",
			PointInteractionType::Other(OtherPointType::Maxima) => "Maxima",
		}
	}
}
#[derive(Clone, Debug)]
pub enum OtherPointType {
	Point,
	Minima,
	Maxima,
	Intersection,
	IntersectionWithXAxis,
	IntersectionWithYAxis,
}
#[derive(Clone, Debug)]
pub enum PointInteractionType {
	Draggable { i: (Id, u32) },
	Other(OtherPointType),
}
pub struct DrawPoint {
	pub sorting_index: u64,
	pub interaction:   PointInteraction,
	pub points:        egui_plot::Points<'static>,
}

impl Clone for DrawPoint {
	fn clone(&self) -> Self {
		let p_points = &self.points as &dyn PlotItem;
		let points = match self.points.geometry() {
			PlotGeometry::Points(plot_points) => plot_points.to_vec(),
			PlotGeometry::None | PlotGeometry::Rects => unreachable!(),
		};

		Self {
			sorting_index: self.sorting_index,
			interaction:   self.interaction.clone(),
			points:        egui_plot::Points::new(p_points.name(), PlotPoints::Owned(points))
				.id(p_points.id())
				.color(p_points.color())
				.color(p_points.color())
				.radius(self.interaction.radius),
		}
	}
}
impl DrawPoint {
	pub fn new(i1: u32, i2: u32, selectable: PointInteraction, points: egui_plot::Points<'static>) -> Self {
		Self { sorting_index: ((i1 as u64) << 32) | i2 as u64, interaction: selectable, points }
	}
}
pub struct DrawPolygonGroup {
	pub sorting_index: u32,
	pub polygons:      Vec<egui_plot::Polygon<'static>>,
	pub fill_color:    Color32,
	stroke:            Stroke,
}

impl Clone for DrawPolygonGroup {
	fn clone(&self) -> Self {
		Self {
			sorting_index: self.sorting_index,
			fill_color:    self.fill_color,
			stroke:        self.stroke,
			polygons:      self
				.polygons
				.iter()
				.map(|p| {
					let p_p = p as &dyn PlotItem;
					let points = match p.geometry() {
						PlotGeometry::None | PlotGeometry::Rects => panic!(),
						PlotGeometry::Points(plot_points) => plot_points.to_vec(),
					};

					egui_plot::Polygon::new(p_p.name(), PlotPoints::Owned(points))
						.id(p_p.id())
						.fill_color(self.fill_color)
						.allow_hover(p_p.allow_hover())
						.stroke(self.stroke)
				})
				.collect(),
		}
	}
}

impl DrawPolygonGroup {
	pub fn new(
		sorting_index: u32, fill_color: Color32, stroke: Stroke, polygons: Vec<egui_plot::Polygon<'static>>,
	) -> Self {
		Self { sorting_index, polygons, fill_color, stroke }
	}
}
#[derive(Clone)]
pub struct DrawText {
	pub text: TextPlotItem,
}
impl DrawText {
	pub fn new(text: TextPlotItem) -> Self { Self { text } }
}

struct ScheduledWork {
	work:      Vec<Box<dyn FnOnce() -> ExecutionResult2 + Send>>,
	timestamp: Instant,
}
struct FinishedWork {
	duration:  Duration,
	timestamp: Instant,
	result:    MultiExecutionResult,
}
pub type ExecutionResult2 = Result<(u64, DrawBuffer), (u64, String)>;
pub type MultiExecutionResult = Vec<Result<(u64, DrawBuffer), (u64, String)>>;
pub struct MultiDrawBufferScheduler {
	pub scheduled:   bool,
	pub deferred:    Option<ScheduledWork>,
	pub average:     (Duration, usize),
	latest_received: Option<Instant>,

	sx: mpsc::SyncSender<FinishedWork>,
	rx: mpsc::Receiver<FinishedWork>,
}
impl MultiDrawBufferScheduler {
	pub fn new() -> Self {
		let (sx, rx) = mpsc::sync_channel(4);
		Self { scheduled: false, average: (Duration::ZERO, 0), latest_received: None, deferred: None, sx, rx }
	}
	pub fn schedule(&mut self, work: Vec<Box<dyn FnOnce() -> ExecutionResult2 + Send + 'static>>) {
		let to_schedule = ScheduledWork { work, timestamp: Instant::now() };

		if !self.scheduled {
			self.spawn_as_latest(to_schedule);
		} else {
			self.deferred = Some(to_schedule);
		}
	}
	fn spawn_as_latest(&mut self, scheduled: ScheduledWork) {
		let sx = self.sx.clone();

		rayon::spawn(move || {
			let start = Instant::now();
			let result = scheduled
				.work
				.into_par_iter()
				.map(|work| {
					let result = work();
					result
				})
				.collect::<Vec<_>>();

			let duration = start.elapsed();

			let _send_res = sx.send(FinishedWork { duration, timestamp: scheduled.timestamp, result });
		});

		self.scheduled = true;
	}
	/// returns (has_outstanding, maybe_new_result)
	pub fn try_receive(&mut self) -> (bool, Option<Vec<Result<(u64, DrawBuffer), (u64, String)>>>) {
		// let mut ido = None;

		let mut result = None;
		while let Ok(finished_work) = self.rx.try_recv() {
			self.scheduled = false;
			self.average.1 += 1;
			self.average.0 += finished_work.duration;

			if let Some(latest) = self.latest_received {
				if finished_work.timestamp < latest {
					continue;
				}
			}
			self.latest_received = Some(finished_work.timestamp);
			result = Some(finished_work.result);
		}

		let has_outstanding = self.scheduled;
		(has_outstanding, result)
	}

	pub fn schedule_deffered_if_idle(&mut self) -> bool {
		if !self.scheduled {
			if let Some(work) = self.deferred.take() {
				self.spawn_as_latest(work);
				return true;
			}
		}
		false
	}
}
