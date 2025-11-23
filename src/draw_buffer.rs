use core::cell::RefCell;
use std::sync::Mutex;

use eframe::egui::{Color32, Id, Mesh, Shape};
use egui_plot::{PlotGeometry, PlotItem, PlotPoint, Points};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use thread_local::ThreadLocal;

use crate::marching_squares::MeshBuilder;
use crate::math::{closest_point_on_segment, dist_sq, intersect_segs};
use crate::thread_local_get;

#[repr(align(128))]
pub struct DrawBufferRC {
	pub inner: RefCell<DrawBuffer>,
}
impl Default for DrawBufferRC {
	fn default() -> Self { Self { inner: RefCell::new(DrawBuffer::default()) } }
}

pub struct DrawBuffer {
	pub lines:    Vec<DrawLine>,
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

pub struct ProcessedDrawBuffers {
	pub closest_point_to_mouse: Option<((f64, f64), f64)>,
	pub draw_lines:             Vec<DrawLine>,
	pub draw_points:            Vec<DrawPoint>,
	pub draw_polygons:          Vec<DrawPolygonGroup>,
	pub draw_texts:             Vec<DrawText>,
	pub draw_meshes:            Vec<DrawMesh>,
}
pub fn process_draw_buffers(
	draw_buffers: &mut ThreadLocal<DrawBufferRC>, selected_plot_line: Option<(Id, bool)>,
	mouse_pos_in_graph: Option<(f64, f64)>, plot_params: &crate::entry::PlotParams,
) -> ProcessedDrawBuffers {
	let mut draw_lines = vec![];
	for draw_buffer in draw_buffers.iter_mut() {
		let draw_buffer = draw_buffer.inner.get_mut();
		draw_lines.append(&mut draw_buffer.lines);
	}
	let final_closest_point_to_mouse: Mutex<Option<((f64, f64), f64)>> = Mutex::new(None);

	if let Some((selected_fline_id, show_closest_point_to_mouse)) = selected_plot_line {
		let mut selected_lines = vec![];
		for (i, line) in draw_lines.iter().enumerate() {
			if line.id == selected_fline_id {
				selected_lines.push(i as u32);
			}
		}

		draw_lines.par_iter().for_each(|fline| {
			let PlotGeometry::Points(plot_points) = fline.line.geometry() else {
				return;
			};
			let mut draw_buffer = thread_local_get(draw_buffers).inner.borrow_mut();
			if fline.id == selected_fline_id {
				let mut closest_point_to_mouse: Option<((f64, f64), f64)> = None;

				// Find local optima
				let mut prev: [Option<(f64, f64)>; 2] = [None; 2];
				for (pi, point) in plot_points.iter().enumerate() {
					let cur = (point.x, point.y);
					fn less_then(a: f64, b: f64, e: f64) -> bool { b - a > e }
					fn greater_then(a: f64, b: f64, e: f64) -> bool { a - b > e }
					if let (Some(prev_0), Some(prev_1)) = (prev[0], prev[1]) {
						// Find closest point to mouse
						if show_closest_point_to_mouse && let Some(mouse_pos_in_graph) = mouse_pos_in_graph {
							let mouse_on_seg = closest_point_on_segment(
								(prev_0.0, prev_0.1),
								(prev_1.0, prev_1.1),
								mouse_pos_in_graph,
							);
							let dist_sq = dist_sq(mouse_on_seg, mouse_pos_in_graph);
							if let Some(cur_closest_point_to_mouse) = closest_point_to_mouse {
								if dist_sq < cur_closest_point_to_mouse.1 {
									closest_point_to_mouse = Some((mouse_on_seg, dist_sq));
								}
							} else {
								closest_point_to_mouse = Some((mouse_on_seg, dist_sq));
							}
						}

						if prev_0.1.signum() != prev_1.1.signum() {
							// intersection with x axis
							let sum = prev_0.1.abs() + prev_1.1.abs();
							let x = if sum == 0.0 {
								(prev_0.0 + prev_1.0) * 0.5
							} else {
								let t = prev_0.1.abs() / sum;
								prev_0.0 + t * (prev_1.0 - prev_0.0)
							};
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x,
									y: 0.0,
									radius: fline.width,
									ty: PointInteractionType::Other(OtherPointType::IntersectionWithXAxis),
								},
								Points::new("", [x, 0.0]).color(Color32::GRAY).radius(fline.width),
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
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x: 0.0,
									y,
									radius: fline.width,
									ty: PointInteractionType::Other(OtherPointType::IntersectionWithYAxis),
								},
								Points::new("", [0.0, y]).color(Color32::GRAY).radius(fline.width),
							));
						}

						if less_then(prev_0.1, prev_1.1, plot_params.eps)
							&& greater_then(prev_1.1, cur.1, plot_params.eps)
						{
							// local maximum
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x:      prev_1.0,
									y:      prev_1.1,
									radius: fline.width,
									ty:     PointInteractionType::Other(OtherPointType::Maxima),
								},
								Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(fline.width),
							));
						}
						if greater_then(prev_0.1, prev_1.1, plot_params.eps)
							&& less_then(prev_1.1, cur.1, plot_params.eps)
						{
							// local minimum
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x:      prev_1.0,
									y:      prev_1.1,
									radius: fline.width,
									ty:     PointInteractionType::Other(OtherPointType::Minima),
								},
								Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(fline.width),
							));
						}
					}

					prev[0] = prev[1];
					prev[1] = Some(cur);
				}
				if show_closest_point_to_mouse {
					if let Some((closest_point, dist_sq)) = closest_point_to_mouse {
						let mut final_closest_point_to_mouse = final_closest_point_to_mouse.lock().unwrap();
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
				// find intersections
				let mut pi = 0;
				for selected_line_i in selected_lines.iter() {
					let selected_line = &draw_lines[*selected_line_i as usize];
					let PlotGeometry::Points(sel_points) = selected_line.line.geometry() else {
						continue;
					};
					for plot_seg in plot_points.windows(2) {
						for sel_seg in sel_points.windows(2) {
							if let Some(point) = intersect_segs(
								plot_seg[0], plot_seg[1], sel_seg[0], sel_seg[1], plot_params.eps,
							) {
								draw_buffer.points.push(DrawPoint::new(
									fline.sorting_index,
									pi as u32,
									PointInteraction {
										x:      point.x,
										y:      point.y,
										radius: fline.width,
										ty:     PointInteractionType::Other(OtherPointType::Intersection),
									},
									Points::new("", [point.x, point.y])
										.color(Color32::GRAY)
										.radius(selected_line.width),
								));
								pi += 1;
							}
						}
					}
				}
			}
		});
	}

	let mut draw_meshes = vec![];
	let mut draw_points = vec![];
	let mut draw_polygons = vec![];
	let mut draw_texts = vec![];
	for draw_buffer in draw_buffers.iter_mut() {
		let draw_buffer = draw_buffer.inner.get_mut();
		draw_polygons.append(&mut draw_buffer.polygons);
		draw_points.append(&mut draw_buffer.points);
		draw_texts.append(&mut draw_buffer.texts);
		draw_meshes.append(&mut draw_buffer.meshes);
	}

	let closest_point_to_mouse = final_closest_point_to_mouse.into_inner().unwrap();
	if let Some(closest_point_to_mouse) = closest_point_to_mouse {
		draw_points.push(DrawPoint::new(
			0,
			0,
			PointInteraction {
				x:      closest_point_to_mouse.0.0,
				y:      closest_point_to_mouse.0.1,
				radius: 5.0,
				ty:     PointInteractionType::Other(OtherPointType::Point),
			},
			Points::new("", [closest_point_to_mouse.0.0, closest_point_to_mouse.0.1])
				.color(Color32::GRAY)
				.radius(5.0),
		));
	}

	draw_lines.sort_unstable_by_key(|draw_line| draw_line.sorting_index);
	draw_points.sort_unstable_by_key(|draw_point| draw_point.sorting_index);
	draw_polygons.sort_unstable_by_key(|draw_poly_group| draw_poly_group.sorting_index);
	draw_texts.sort_unstable_by_key(|draw_text| draw_text.sorting_index);
	draw_meshes.sort_unstable_by_key(|draw_mesh| draw_mesh.sorting_index);

	ProcessedDrawBuffers {
		closest_point_to_mouse,
		draw_lines,
		draw_points,
		draw_polygons,
		draw_texts,
		draw_meshes,
	}
}

pub struct DrawMesh {
  pub bounds:egui_plot::PlotBounds,
  pub plot_item_base: egui_plot::PlotItemBase,
	pub sorting_index: u32,
	pub mesh:          RefCell<MeshBuilder>,
  pub color: Color32,
}
impl PlotItem for DrawMesh {
    fn shapes(&self, ui: &eframe::egui::Ui, transform: &egui_plot::PlotTransform, shapes: &mut Vec<Shape>) {
      for vertex in self.mesh.borrow_mut().vertices.iter_mut(){
        vertex.pos = transform.position_from_point(&PlotPoint::new(vertex.pos.x as f64, vertex.pos.y as f64));

      }
      let mesh = core::mem::take(&mut *self.mesh.borrow_mut());
      let mesh = Mesh{
        vertices: mesh.vertices,
        indices: mesh.indices,
        ..Default::default()
      };

      shapes.push(Shape::mesh(mesh));
    }

    fn initialize(&mut self, x_range: std::ops::RangeInclusive<f64>) {
    }

    fn color(&self) -> Color32 {
      self.color
    }

    fn geometry(&self) -> PlotGeometry<'_> {
      PlotGeometry::None
    }

    fn bounds(&self) -> egui_plot::PlotBounds {
      self.bounds
    }

    fn base(&self) -> &egui_plot::PlotItemBase {
      &self.plot_item_base
    }

    fn base_mut(&mut self) -> &mut egui_plot::PlotItemBase {
      &mut self.plot_item_base
    }
}
pub struct DrawLine {
	pub sorting_index: u32,
	pub line:          egui_plot::Line<'static>,
	pub id:            Id,
	pub width:         f32,
}
impl DrawLine {
	pub fn new(sorting_index: u32, id: Id, width: f32, line: egui_plot::Line<'static>) -> Self {
		Self { sorting_index, id, width, line }
	}
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
impl DrawPoint {
	pub fn new(i1: u32, i2: u32, selectable: PointInteraction, points: egui_plot::Points<'static>) -> Self {
		Self { sorting_index: ((i1 as u64) << 32) | i2 as u64, interaction: selectable, points }
	}
}
pub struct DrawPolygonGroup {
	pub sorting_index: u32,
	pub polygons:      Vec<egui_plot::Polygon<'static>>,
}
impl DrawPolygonGroup {
	pub fn new(sorting_index: u32, polygons: Vec<egui_plot::Polygon<'static>>) -> Self {
		Self { sorting_index, polygons }
	}
}
pub struct DrawText {
	pub sorting_index: u32,
	pub text:          egui_plot::Text,
}
impl DrawText {
	pub fn new(sorting_index: u32, text: egui_plot::Text) -> Self { Self { sorting_index, text } }
}

/// SAFETY: Not Sync because of `ExplicitGenerator` callbacks, but we dont use those.
unsafe impl Sync for DrawLine {}
