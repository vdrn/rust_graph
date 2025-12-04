use alloc::sync::Arc;
use core::cell::{RefCell, RefMut};
use core::mem;
use core::sync::atomic::AtomicBool;
use std::marker::ConstParamTy;

use eframe::egui::{self, Align2, Color32, Id, Pos2, RichText, Stroke, pos2, remap};
use egui_plot::{Line, PlotItemBase, PlotPoint, PlotPoints, PlotTransform, Points, Polygon};
use evalexpr::{EvalexprError, EvalexprFloat, ExpressionFunction, FlatNode, HashMapContext, Stack, Value};
use thread_local::ThreadLocal;

use crate::draw_buffer::{
	DrawBuffer, DrawLine, DrawMesh, DrawMeshType, DrawPoint, DrawPolygonGroup, DrawText, EguiPlotMesh, ExecutionResult, FillMesh, OtherPointType, PointInteraction, PointInteractionType
};
use crate::entry::{
	COLORS, ClonedEntry, DragPoint, Entry, EntryType, EquationType, NUM_COLORS, PointsType, f64_to_value
};
use crate::marching_squares::MeshBuilder;
use crate::math::{
	DiscontinuityDetector, aabb_segment_intersects_loose, pseudoangle, zoom_in_x_on_nan_boundary
};
use crate::widgets::TextPlotItem;
use crate::{ThreadLocalContext, UiState, marching_squares, thread_local_get};

#[derive(Clone)]
pub struct PlotParams {
	pub eps:                 f64,
	pub first_x:             f64,
	pub last_x:              f64,
	pub first_y:             f64,
	pub last_y:              f64,
	pub step_size:           f64,
	pub step_size_y:         f64,
	pub resolution:          usize,
	pub prev_plot_transform: Option<egui_plot::PlotTransform>,
	pub invert_axes:         [bool; 2],
}
impl PlotParams {
	pub fn new(ui_state: &UiState) -> Self {
		let first_x = ui_state.plot_bounds.min()[0];
		let last_x = ui_state.plot_bounds.max()[0];
		let first_y = ui_state.plot_bounds.min()[1];
		let last_y = ui_state.plot_bounds.max()[1];

		// let (first_x, last_x) = snap_range_to_grid(first_x, last_x, 10.0);
		let plot_width = last_x - first_x;
		let plot_height = last_y - first_y;

		let mut points_to_draw = ui_state.conf.resolution.max(1);
		let mut step_size = plot_width / points_to_draw as f64;
		while points_to_draw > 2 && first_x + step_size == first_x {
			points_to_draw /= 2;
			step_size = plot_width / points_to_draw as f64;
		}

		let mut points_to_draw_y = ui_state.conf.resolution.max(1);
		let mut step_size_y = plot_height / points_to_draw as f64;
		while points_to_draw_y > 2 && first_y + step_size_y == first_y {
			points_to_draw_y /= 2;
			step_size_y = plot_height / points_to_draw_y as f64;
		}
		Self {
			eps: if ui_state.conf.use_f32 { ui_state.f32_epsilon } else { ui_state.f64_epsilon },
			first_x,
			last_x,
			first_y,
			last_y,
			step_size,
			step_size_y,
			resolution: ui_state.conf.resolution,
			prev_plot_transform: ui_state.prev_plot_transform,
			invert_axes: ui_state.graph_config.invert_axes,
		}
	}
}
pub fn schedule_entry_create_plot_elements<T: EvalexprFloat>(
	entry: &mut Entry<T>, sorting_idx: u32, selected_id: Option<Id>, ctx: &Arc<evalexpr::HashMapContext<T>>,
	plot_params: &PlotParams, tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>,
) {
	let visible = entry.active;
	if !visible && !matches!(entry.ty, EntryType::Folder { .. }) {
		entry.draw_buffer_scheduler.clear_buffer();
		return;
	}
	match &mut entry.ty {
		EntryType::Constant { .. } => {},
		EntryType::Folder { entries } => {
			for (ei, entry) in entries.iter_mut().enumerate() {
				schedule_entry_create_plot_elements(
					entry,
					sorting_idx + ei as u32,
					selected_id,
					ctx,
					plot_params,
					tl_context,
				);
			}
		},
		EntryType::Function { can_be_drawn, .. } => {
			if !*can_be_drawn {
				entry.draw_buffer_scheduler.clear_buffer();
			} else {
				entry.draw_buffer_scheduler.schedule(entry.id, {
					let entry_cloned =
						ClonedEntry { id: entry.id, color: entry.color, ty: entry.ty.clone() };
					let ctx = Arc::clone(ctx);
					let plot_params = plot_params.clone();
					let tl_context = Arc::clone(tl_context);

					move || {
						entry_create_plot_elements_async(
							&entry_cloned, sorting_idx, &ctx, &plot_params, &tl_context,
						)
					}
				});
			}
		},

		_ => {
			entry.draw_buffer_scheduler.execute({
				|draw_buffer| {
					entry_create_plot_elements_sync(
						entry.id, &entry.ty, &entry.name, entry.color, sorting_idx, selected_id, ctx,
						plot_params, tl_context, draw_buffer,
					);
					Ok(())
				}
			});
		},
	}
}
pub fn entry_create_plot_elements_async<T: EvalexprFloat>(
	entry: &ClonedEntry<T>, sorting_idx: u32, ctx: &evalexpr::HashMapContext<T>, plot_params: &PlotParams,
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>,
) -> ExecutionResult {
	let color = entry.color();
	let id = Id::new(entry.id);

	let mut draw_buffer = DrawBuffer::empty();
	match &entry.ty {
		EntryType::Folder { .. } | EntryType::Constant { .. } | EntryType::Points { .. } => {
			unreachable!()
		},
		EntryType::Function {
			func,
			identifier,
			parametric,
			parametric_fill,
			range_start,
			range_end,
			style,
			fill_rule,
			implicit_resolution,
			..
		} => {
			let stack_len = thread_local_get(tl_context).stack.borrow().len();
			assert!(stack_len == 0, "stack is not empty: {stack_len}");

			let display_name = if identifier.is_empty() {
				format!("f({}): {}", func.args_to_string(), func.text.trim())
			} else {
				format!("{}({}): {}", identifier, func.args_to_string(), func.text.trim())
			};

			// let selected = selected_id == Some(id);
			// let width = if selected { style.line_width + 2.5 } else { style.line_width };
			let width = style.line_width;

			let mut add_line = |line: Vec<PlotPoint>| {
				if line.is_empty() {
					return;
				}
				draw_buffer.lines.push(DrawLine::new(
					sorting_idx,
					id,
					width,
					style.egui_line_style(),
					Line::new(&display_name, PlotPoints::Owned(line))
						.id(id)
						.allow_hover(style.selectable)
						.width(width)
						.style(style.egui_line_style())
						.color(color),
				));
			};
			let mut add_egui_plot_mesh = |mesh: MeshBuilder| {
				if mesh.is_empty() {
					return;
				}
				draw_buffer.meshes.push(DrawMesh {
					ty: DrawMeshType::EguiPlotMesh(EguiPlotMesh {
						bounds: mesh.bounds,
						mesh: RefCell::new(mesh),
						color,
						plot_item_base: PlotItemBase::new(display_name.clone()),
					}),
				});
			};
			let fill_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 128);

			if let Some(expr_func) = &func.expr_function {
				if func.args.is_empty() {
					let mut stack = thread_local_get(tl_context).stack.borrow_mut();
					// horizontal line
					let value = match expr_func.call(&mut stack, ctx, &[]) {
						Ok(v) => match v {
							Value::Float(v) => v,
							Value::Float2(_, _) | Value::Boolean(_) | Value::Tuple(_) | Value::Empty => {
								return Ok(draw_buffer);
							},
						},
						Err(e) => return Err((entry.id, e.to_string())),
					};
					add_line(vec![
						PlotPoint::new(plot_params.first_x, value.to_f64()),
						PlotPoint::new(plot_params.last_x, value.to_f64()),
					]);
				} else if func.args.len() == 1 {
					// starndard X or Y function
					match func.equation_type {
						EquationType::None => {
							if *parametric {
								let mut fill_mesh = if *parametric_fill
									&& let Some(plot_trans) = plot_params.prev_plot_transform
								{
									Some((FillMesh::new(fill_color, *fill_rule), plot_trans))
								} else {
									None
								};

								let mut add_parametric_line = |line: Vec<PlotPoint>, break_fill_mesh: bool| {
									if line.is_empty() && !break_fill_mesh {
										return;
									}
									if let Some((fm, plot_trans)) = &mut fill_mesh {
										for point in line.iter() {
											let screen_point = gpu_position_from_point(
												plot_trans, plot_params.invert_axes, point,
											);
											fm.add_vertex(screen_point.x, screen_point.y);
										}

										if break_fill_mesh {
											fm.reset_root_vertex();
											// let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
											// draw_buffer.meshes.push(DrawMesh {
											// 	sorting_index: sorting_idx,
											// 	ty:            DrawMeshType::FillMesh(mem::take(fm)),
											// });
											// *fm = FillMesh::new(fill_color, *fill_rule);
										}
									}

									add_line(line);
								};

								if func.args[0].to_str() == "y" {
									draw_parametric_function::<T, { SimpleFunctionType::Y }>(
										tl_context,
										ctx,
										plot_params,
										entry.id,
										expr_func,
										range_start.node.as_ref(),
										range_end.node.as_ref(),
										&mut add_parametric_line,
									)?;
								} else {
									draw_parametric_function::<T, { SimpleFunctionType::X }>(
										tl_context,
										ctx,
										plot_params,
										entry.id,
										expr_func,
										range_start.node.as_ref(),
										range_end.node.as_ref(),
										&mut add_parametric_line,
									)?;
								}

								if let Some((fm, _)) = fill_mesh {
									draw_buffer.meshes.push(DrawMesh { ty: DrawMeshType::FillMesh(fm) });
								}
							} else {
								if func.args[0].to_str() == "y" {
									draw_simple_function::<T, { SimpleFunctionType::Y }>(
										tl_context, ctx, plot_params, entry.id, expr_func, &mut add_line,
									)?;
								} else {
									draw_simple_function::<T, { SimpleFunctionType::X }>(
										tl_context, ctx, plot_params, entry.id, expr_func, &mut add_line,
									)?;
								}
							}
						},
						EquationType::Equality => {
							let mut stack = Stack::<T>::with_capacity(0);
							let value = match expr_func.call(&mut stack, ctx, &[Value::Float(T::ZERO)]) {
								Ok(v) => match v {
									Value::Float(v) => v,
									Value::Float2(_, _)
									| Value::Boolean(_)
									| Value::Tuple(_)
									| Value::Empty => {
										return Ok(draw_buffer);
									},
								},
								Err(e) => return Err((entry.id, e.to_string())),
							};

							if func.args[0].to_str() == "x" {
								add_line(vec![
									PlotPoint::new(-value.to_f64(), plot_params.first_y),
									PlotPoint::new(-value.to_f64(), plot_params.last_y),
								]);
							} else {
								add_line(vec![
									PlotPoint::new(plot_params.first_x, -value.to_f64()),
									PlotPoint::new(plot_params.last_x, -value.to_f64()),
								]);
							}
						},
						equation_type => {
							if func.args[0].to_str() == "y" {
								draw_implicit(
									tl_context,
									plot_params,
									entry.id,
									*implicit_resolution,
									equation_type,
									fill_color,
									|cc, _, y| expr_func.call(cc, ctx, &[f64_to_value::<T>(y)]),
									&mut add_line,
									&mut add_egui_plot_mesh,
								)?;
							} else {
								draw_implicit(
									tl_context,
									plot_params,
									entry.id,
									*implicit_resolution,
									equation_type,
									fill_color,
									|cc, x, _| expr_func.call(cc, ctx, &[f64_to_value::<T>(x)]),
									&mut add_line,
									&mut add_egui_plot_mesh,
								)?;
							}
						},
					}
				} else if func.args.len() == 2 {
					if func.args[0].to_str() == "x" && func.args[1].to_str() == "y" {
						if func.equation_type != EquationType::None {
							draw_implicit(
								tl_context,
								plot_params,
								entry.id,
								*implicit_resolution,
								func.equation_type,
								fill_color,
								|cc, x, y| {
									expr_func.call(cc, ctx, &[f64_to_value::<T>(x), f64_to_value::<T>(y)])
								},
								&mut add_line,
								&mut add_egui_plot_mesh,
							)?;
						}
					}
				}
			}
		},
	}
	Ok(draw_buffer)
}
pub fn entry_create_plot_elements_sync<T: EvalexprFloat>(
	id: u64, ty: &EntryType<T>, name: &str, color: usize, sorting_idx: u32, selected_id: Option<Id>,
	ctx: &evalexpr::HashMapContext<T>, plot_params: &PlotParams,
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, draw_buffer: &mut DrawBuffer,
) {
	let color = COLORS[color % NUM_COLORS];
	// println!("step_size: {deriv_step_x}");
	let egui_id = Id::new(id);

	match ty {
		EntryType::Folder { .. } | EntryType::Constant { .. } | EntryType::Function { .. } => {
			unreachable!()
		},
		EntryType::Points { points_ty, style, .. } => {
			// main_context
			// 	.write()
			// 	.unwrap()
			// 	.set_value("x", evalexpr::Value::<T>::Float(T::ZERO))
			// 	.unwrap();
			let mut arrow_buffer = vec![];
			let mut line_buffer = vec![];
			let color_rgba = color.to_array();
			let color_outer =
				Color32::from_rgba_unmultiplied(color_rgba[0], color_rgba[1], color_rgba[1], 128);
			let mut prev_point: Option<egui::Vec2> = None;
			let arrow_scale = egui::Vec2::new(
				(plot_params.last_x - plot_params.first_x) as f32,
				(plot_params.last_y - plot_params.first_y) as f32,
			) * 0.002;
			let (points_len, points_iter): (
				usize,
				Box<dyn Iterator<Item = (usize, Option<((T, T), Option<DragPoint>)>)>>,
			) = match points_ty {
				PointsType::Separate(points) => (
					points.len(),
					Box::new(points.iter().map(|p| p.val.map(|v| (v, p.drag.drag_point.clone()))).enumerate()),
				),
				PointsType::SingleExpr { expr: _, val } => {
					(val.len(), Box::new(val.iter().map(|p| Some((*p, None))).enumerate()))
				},
			};

			let mut fill_mesh = if style.fill
				&& points_len > 2
				&& let Some(plot_trans) = &plot_params.prev_plot_transform
			{
				let fill_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 128);
				Some((FillMesh::new(fill_color, style.fill_rule), plot_trans))
			} else {
				None
			};
			for (i, p) in points_iter {
				if let Some(((x, y), drag_point)) = p {
					let x = x.to_f64();
					let y = y.to_f64();
					let point_id = egui_id.with(i);
					let selected = selected_id == Some(egui_id);
					let radius = if selected { 6.5 } else { 4.5 };
					let radius_outer = if selected { 12.5 } else { 7.5 };

					if let Some((fill_mesh, plot_trans)) = &mut fill_mesh {
						let screen_point = gpu_position_from_point(
							plot_trans,
							plot_params.invert_axes,
							&PlotPoint::new(x, y),
						);
						fill_mesh.add_vertex(screen_point.x, screen_point.y);
					}
					if style.show_points || drag_point.is_some() {
						draw_buffer.points.push(DrawPoint::new(
							sorting_idx,
							i as u32,
							PointInteraction {
								x,
								y,
								radius,
								ty: PointInteractionType::Other(OtherPointType::Point),
							},
							Points::new(name.to_string(), [x, y]).color(color).radius(radius),
						));
						if drag_point.is_some() {
							let selectable_point = PointInteraction {
								ty: PointInteractionType::Draggable { i: (egui_id, i as u32) },
								x,
								y,
								radius: radius_outer,
							};
							draw_buffer.points.push(DrawPoint::new(
								sorting_idx,
								i as u32,
								selectable_point,
								Points::new(name.to_string(), [x, y])
									.id(point_id)
									.color(color_outer)
									.radius(radius_outer),
							));
						}
					}
					if let Some(label_config) = &style.label_config
						&& i == points_len - 1
						&& !label_config.text.trim().is_empty()
					{
						let mut stack = thread_local_get(tl_context).stack.borrow_mut();
						let angle = label_config
							.angle
							.node
							.as_ref()
							.and_then(|expr| expr.eval_float_with_context(&mut stack, ctx).ok())
							.map(|angle| angle.to_f64() as f32)
							.unwrap_or(0.0);
						let size = label_config.size.size();
						let text = label_config.text.trim();
						let mut label = RichText::new(text).size(size);
						if label_config.italic {
							label = label.italics();
						}

						let dir = label_config.pos.dir();
						let text = TextPlotItem::new(
							text,
							PlotPoint {
								x: x + (dir.x * size * arrow_scale.x * 0.8) as f64,
								y: y + (dir.y * size * arrow_scale.y * 0.8) as f64,
							},
							label,
						)
						.with_angle(angle, Align2::CENTER_CENTER)
						.with_color(color);

						draw_buffer.texts.push(DrawText::new(text));
					}

					if style.show_lines {
						line_buffer.push([x, y]);
						let cur_point = egui::Vec2::new(x as f32, y as f32);
						if style.show_arrows
							&& let Some(pp) = prev_point
						{
							let dir = (pp - cur_point).normalized();
							let arrow_len = arrow_scale * radius_outer;
							let base = cur_point + dir * arrow_len.length();
							let a = base + dir.rot90() * arrow_len * 0.5;
							let b = base - dir.rot90() * arrow_len * 0.5;

							arrow_buffer.push(
								Polygon::new(
									"",
									vec![[x, y], [a.x as f64, a.y as f64], [b.x as f64, b.y as f64]],
								)
								.fill_color(color)
								.allow_hover(false)
								.stroke(Stroke::new(0.0, color)),
							);
						}
						prev_point = Some(cur_point);
					}
				}
			}

			// let width = if selected { 3.5 } else { 1.0 };
			let width = 1.0;

			if line_buffer.len() > 1 && style.show_lines {
				if style.connect_first_and_last {
					line_buffer.push(line_buffer[0]);
				}
				let line = Line::new(name.to_string(), line_buffer)
					.color(color)
					.id(egui_id)
					.width(style.line_style.line_width)
					.allow_hover(style.line_style.selectable)
					.style(style.line_style.egui_line_style());
				draw_buffer.lines.push(DrawLine::new(
					sorting_idx,
					egui_id,
					width,
					style.line_style.egui_line_style(),
					line,
				));
				if !arrow_buffer.is_empty() {
					draw_buffer.polygons.push(DrawPolygonGroup::new(
						sorting_idx,
						color,
						Stroke::new(0.0, color),
						arrow_buffer,
					));
				}
				if let Some((fill_mesh, _)) = fill_mesh {
					draw_buffer.meshes.push(DrawMesh { ty: DrawMeshType::FillMesh(fill_mesh) });
				}
				// plot_ui.line(line);
			}
		},
	}
}

pub fn eval_point2<T: EvalexprFloat>(
	stack: &mut Stack<T>, ctx: &evalexpr::HashMapContext<T>, px: Option<&FlatNode<T>>,
	py: Option<&FlatNode<T>>,
) -> Result<Option<(T, T)>, String> {
	let (Some(x), Some(y)) = (px, py) else {
		return Ok(None);
	};
	let x = x.eval_float_with_context(stack, ctx).map_err(|e| e.to_string())?;
	let y = y.eval_float_with_context(stack, ctx).map_err(|e| e.to_string())?;
	Ok(Some((x, y)))
}
pub fn eval_point<T: EvalexprFloat>(
	stack: &mut Stack<T>, ctx: &evalexpr::HashMapContext<T>, px: Option<&FlatNode<T>>,
	py: Option<&FlatNode<T>>,
) -> Result<Option<(f64, f64)>, String> {
	let (Some(x), Some(y)) = (px, py) else {
		return Ok(None);
	};
	let x = x.eval_float_with_context(stack, ctx).map_err(|e| e.to_string())?;
	let y = y.eval_float_with_context(stack, ctx).map_err(|e| e.to_string())?;
	Ok(Some((x.to_f64(), y.to_f64())))
}

#[derive(PartialEq, Eq, ConstParamTy)]
enum SimpleFunctionType {
	X,
	Y,
}
fn draw_simple_function<T: EvalexprFloat, const TY: SimpleFunctionType>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, ctx: &HashMapContext<T>, plot_params: &PlotParams,
	id: u64, func: &ExpressionFunction<T>, mut add_line: impl FnMut(Vec<PlotPoint>),
) -> Result<(), (u64, String)> {
	let mut stack = thread_local_get(tl_context).stack.borrow_mut();
	let mut pp_buffer = vec![];
	let mut prev_sampling_point: Option<(f64, f64)> = None;

	let mut sampling_arg = match TY {
		SimpleFunctionType::X => plot_params.first_x,
		SimpleFunctionType::Y => plot_params.first_y,
	};
	let last_sampling_arg = match TY {
		SimpleFunctionType::X => plot_params.last_x,
		SimpleFunctionType::Y => plot_params.last_y,
	};
	let step_size = match TY {
		SimpleFunctionType::X => plot_params.step_size,
		SimpleFunctionType::Y => plot_params.step_size_y,
	};

	if let Some(constant) = func.as_constant() {
		let value = match constant {
			Value::Float(v) => v,
			Value::Float2(_, _) | Value::Boolean(_) | Value::Tuple(_) | Value::Empty => {
				return Ok(());
			},
		};
		match TY {
			SimpleFunctionType::X => {
				pp_buffer.push(PlotPoint::new(plot_params.first_x, value.to_f64()));
				pp_buffer.push(PlotPoint::new(plot_params.last_x, value.to_f64()));
			},
			SimpleFunctionType::Y => {
				pp_buffer.push(PlotPoint::new(value.to_f64(), plot_params.first_y));
				pp_buffer.push(PlotPoint::new(value.to_f64(), plot_params.last_y));
			},
		}
	} else {
		// let mut prev_y = None;
		let mut discontinuity_detector = DiscontinuityDetector::new(step_size, plot_params.eps);

		// let graph_size = match TY {
		// 	SimpleFunctionType::X => plot_params.last_y - plot_params.first_y,
		// 	SimpleFunctionType::Y => plot_params.last_x - plot_params.first_x,
		// };
		// let mut prev_angle: Option<f64> = None;
		// let mut prev_point: Option<PlotPoint> = None;
		while sampling_arg <= last_sampling_arg {
			match func.call(&mut stack, ctx, &[f64_to_value::<T>(sampling_arg)]) {
				Ok(Value::Float(val)) => {
					let val = val.to_f64();

					let zoomed_in_on_nan_boundary = prev_sampling_point.and_then(|(prev_arg, prev_val)| {
						zoom_in_x_on_nan_boundary(
							(prev_arg, prev_val),
							(sampling_arg, val),
							plot_params.eps,
							|arg| {
								func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)])
									.and_then(|v| v.as_float())
									.map(|v| v.to_f64())
									.ok()
							},
						)
					});

					let mut on_nan_boundary = false;

					let (cur_arg, cur_val) = if let Some(zoomed_in_on_nan_boundary) = zoomed_in_on_nan_boundary
					{
						on_nan_boundary = true;
						zoomed_in_on_nan_boundary
					} else {
						(sampling_arg, val)
					};

					let latest_point = if !on_nan_boundary
						&& let Some((left, right)) = discontinuity_detector.detect(cur_arg, cur_val, |arg| {
							func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)])
								.and_then(|v| v.as_float())
								.map(|v| v.to_f64())
								.ok()
						}) {
						match TY {
							SimpleFunctionType::X => {
								pp_buffer.push(PlotPoint::new(left.0, left.1));
								add_line(mem::take(&mut pp_buffer));
								Some(PlotPoint::new(right.0, right.1))
							},
							SimpleFunctionType::Y => {
								pp_buffer.push(PlotPoint::new(left.1, left.0));
								add_line(mem::take(&mut pp_buffer));
								Some(PlotPoint::new(right.1, right.0))
							},
						}
					} else {
						if !cur_val.is_nan() {
							Some(match TY {
								SimpleFunctionType::X => PlotPoint::new(cur_arg, cur_val),
								SimpleFunctionType::Y => PlotPoint::new(cur_val, cur_arg),
							})
						} else {
							None
						}
					};
					if let Some(latest_point) = latest_point {
						// TODO:This is commented out because we cannot just skip the points,
						// as the egui_plot hovering functionality does not work without dense
						// poitns on lines.
						// We need to implement our hovering for this optimization.
						//
						// if let Some(prev_point) = prev_point {
						// 	let dx = latest_point.x - prev_point.x;
						// 	let dy = latest_point.y - prev_point.y;
						// 	let pangle = pseudoangle(dx, dy);
						// 	if let Some(prev_angle) = prev_angle
						// 		&& pp_buffer.len() > 1
						// 	{
						// 		let angle_diff = (pangle - prev_angle).abs();
						// 		if angle_diff < 0.00001 * graph_size {
						// 			*pp_buffer.last_mut().unwrap() = latest_point;
						// 			// pp_buffer.push(latest_point);
						// 		} else {
						// 			pp_buffer.push(latest_point);
						// 		}
						// 	} else {
						// 		pp_buffer.push(latest_point);
						// 	}
						// 	prev_angle = Some(pangle);
						// }
						// prev_point = Some(latest_point);
						pp_buffer.push(latest_point);
					}

					prev_sampling_point = Some((sampling_arg, val));
				},
				Ok(Value::Empty) => {
					add_line(mem::take(&mut pp_buffer));
				},
				Ok(Value::Tuple(_) | Value::Float2(_, _)) => {
					return Ok(());
					// return Err((id, "Non-parametric function must return a single number.".to_string()));
				},
				Ok(Value::Boolean(_)) => {
					return Ok(());

					// return Err((id, "Non-parametric function must return a number.".to_string()));
				},
				Err(e) => {
					return Err((id, e.to_string()));
				},
			}

			let prev_sampling_arg = sampling_arg;
			sampling_arg += step_size;
			if sampling_arg == prev_sampling_arg {
				break;
			}
		}
	}

	// println!("fn {} num points", pp_buffer.len());
	add_line(pp_buffer);

	Ok(( ))
}

const P_CURVATURE_FACTOR_SCALE: f64 = 8.0;
const P_SMALL_DISTANCE_SCALE: f64 = 18.0;
const P_SMALL_DISTANCE_POW: f64 = 1.22;
fn calculate_curvature_factor(pangle1: f64, pangle2: f64) -> f64 {
	let mut angle_diff = (pangle2 - pangle1).abs();
	if angle_diff > 2.0 {
		angle_diff = 4.0 - angle_diff;
	}

	// Convert to curvature factor: 1.0 = straight, increases with curvature
	1.0 + (angle_diff * P_CURVATURE_FACTOR_SCALE)
}
#[allow(clippy::too_many_arguments)]
fn draw_parametric_function<T: EvalexprFloat, const TY: SimpleFunctionType>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, ctx: &HashMapContext<T>, plot_params: &PlotParams,
	id: u64, func: &ExpressionFunction<T>, range_start: Option<&FlatNode<T>>, range_end: Option<&FlatNode<T>>,
	mut add_line: impl FnMut(Vec<PlotPoint>, bool),
) -> Result<(), (u64, String)> {
	let mut stack = thread_local_get(tl_context).stack.borrow_mut();
	let mut pp_buffer: Vec<PlotPoint> = Vec::with_capacity(plot_params.resolution);
	#[inline(always)]
	fn add_point(
		pp_buffer: &mut Vec<PlotPoint>, p: PlotPoint, plot_params: &PlotParams,
		add_line: &mut impl FnMut(Vec<PlotPoint>, bool),
	) {
		if let Some(last) = pp_buffer.last() {
			let on_screen = aabb_segment_intersects_loose(
				(plot_params.first_x, plot_params.first_y),
				(plot_params.last_x, plot_params.last_y),
				(last.x, last.y),
				(p.x, p.y),
			);
			if !on_screen {
				if pp_buffer.len() > 1 {
					add_line(mem::take(pp_buffer), true);
				} else {
					pp_buffer.clear();
				}
			}
		}
		// if pp_buffer.len() > 1 {
		// 	let last = pp_buffer[pp_buffer.len() - 1];
		// 	let sec_last = pp_buffer[pp_buffer.len() - 2];

		// 	let on_screen = aabb_segment_intersects_loose(
		// 		(plot_params.first_x, plot_params.first_y),
		// 		(plot_params.last_x, plot_params.last_y),
		// 		(last.x, last.y),
		// 		(sec_last.x, sec_last.y),
		// 	);

		// 	if !on_screen {
		// 		let last = pp_buffer.pop().unwrap();
		// 		pp_buffer.pop();
		// 		add_line(mem::take(pp_buffer), true);
		// 		pp_buffer.push(last);
		// 		return;
		// 	}
		// }
		pp_buffer.push(p);
	}
	let (start, end) = match eval_point(&mut stack, ctx, range_start, range_end) {
		Ok(Some((start, end))) => (start, end),
		Err(e) => {
			return Err((id, e));
		},
		_ => {
			return Ok(());
		},
	};
	if start > end {
		return Err((id, "Range start must be less than range end".to_string()));
	}
	let range = end - start;
	let step = range / plot_params.resolution as f64;
	if start + step == end {
		return Ok(());
	}

	let plot_width = plot_params.last_x - plot_params.first_x;
	let plot_height = plot_params.last_y - plot_params.first_y;

	let small_distance_x = (plot_params.eps * 10_000.0).max(plot_width / plot_params.resolution as f64);
	let small_distance_y = (plot_params.eps * 10_000.0).max(plot_height / plot_params.resolution as f64);
	let max_segment_x = small_distance_x.powf(P_SMALL_DISTANCE_POW) * P_SMALL_DISTANCE_SCALE;
	let max_segment_y = small_distance_y.powf(P_SMALL_DISTANCE_POW) * P_SMALL_DISTANCE_SCALE;
	// println!(
	// 	"small distance x: {small_distance_x} plot eps * 10_000 {} max segment x {max_segment_x}",
	// 	plot_params.eps * 10_000.0
	// );

	#[allow(clippy::integer_division)]
	let max_subdivisions = (plot_params.resolution / 20).max(1);

	// Determine function type from first evaluation
	let first_value = match func.call(&mut stack, ctx, &[f64_to_value::<T>(start)]) {
		Ok(v) => v,
		Err(e) => return Err((id, e.to_string())),
	};

	let is_float2 = matches!(first_value, Value::Float2(_, _));

	let step_eps = if is_float2 {
		max_segment_y
	} else if TY == SimpleFunctionType::X {
		max_segment_x
	} else {
		max_segment_y
	};

	let mut eval_point_at = |arg: f64| -> Result<Option<(f64, PlotPoint)>, String> {
		match func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)]) {
			Ok(Value::Float(value)) => {
				if is_float2 {
					return Err("Function must consistently return Float2".to_string());
				}
				if value.is_nan() {
					Ok(None)
				} else {
					let v = value.to_f64();
					match TY {
						SimpleFunctionType::X => Ok(Some((arg, PlotPoint::new(arg, v)))),
						SimpleFunctionType::Y => Ok(Some((arg, PlotPoint::new(v, arg)))),
					}
				}
			},
			Ok(Value::Empty) => Ok(None),
			Ok(Value::Float2(x, y)) => {
				if !is_float2 {
					return Err("Function must consistently return Float".to_string());
				}
				if y.is_nan() {
					Ok(None)
				} else {
					// Use y value for discontinuity detection
					Ok(Some((arg, PlotPoint::new(x.to_f64(), y.to_f64()))))
				}
			},
			Ok(_) => Err("Ranged function must return 1 or 2 float values".to_string()),
			Err(e) => Err(e.to_string()),
		}
	};

	let mut prev_angle: Option<f64> = None;
	// Evaluate first point
	let mut prev_arg = start;
	let mut discontinuity_detector = None;
	let mut discontinuity_detector2 = None;
	let mut prev_point = match eval_point_at(start) {
		Ok(Some((arg, p))) => {
			add_point(&mut pp_buffer, p, plot_params, &mut add_line);
			// println!("Start p {p:?}");
			if is_float2 {
				discontinuity_detector =
					Some(DiscontinuityDetector::new_with_initial(step_eps, plot_params.eps, (arg, p.y)));
				discontinuity_detector2 =
					Some(DiscontinuityDetector::new_with_initial(step_eps, plot_params.eps, (arg, p.x)));
			} else {
				match TY {
					SimpleFunctionType::X => {
						discontinuity_detector = Some(DiscontinuityDetector::new_with_initial(
							step_eps,
							plot_params.eps,
							(p.x, p.y),
						));
					},
					SimpleFunctionType::Y => {
						discontinuity_detector = Some(DiscontinuityDetector::new_with_initial(
							step_eps,
							plot_params.eps,
							(p.y, p.x),
						));
					},
				}
			}

			Some(p)
		},
		Ok(None) => None,
		Err(e) => return Err((id, e)),
	};

	let mut discontinuity_detector =
		discontinuity_detector.unwrap_or_else(|| DiscontinuityDetector::new(step_eps, plot_params.eps));
	let mut discontinuity_detector2 =
		discontinuity_detector2.unwrap_or_else(|| DiscontinuityDetector::new(step_eps, plot_params.eps));

	// let mut num_on_screen = 0;
	// let mut num_off_screen = 0;
	// let mut max_base_subdivisions = 0;
	// let mut num_splits = Vec::with_capacity(plot_params.resolution * 2);
	// let mut curvature_factors = Vec::with_capacity(plot_params.resolution * 2);
	for i in 1..=plot_params.resolution {
		let curr_arg = start + step * i as f64;
		let curr_result = match eval_point_at(curr_arg) {
			Ok(r) => r,
			Err(e) => return Err((id, e)),
		};
		// let debug = i < 3;
		let debug = false;
		if debug {
			println!("i = {i} start p {curr_result:?}");
		}
		let mut cur_angle = None;

		match (prev_point, curr_result) {
			(Some(prev_p), Some((_, curr_p))) => {
				// Check if segment is too long and needs subdivision
				let dx = curr_p.x - prev_p.x;
				let dy = curr_p.y - prev_p.y;
				let a_dx = dx.abs();
				let a_dy = dy.abs();

				let margin_x = plot_width * 0.5;
				let margin_y = plot_height * 0.5;
				// let margin_x = 0.0;
				// let margin_y = 0.0;
				let on_screen = aabb_segment_intersects_loose(
					(plot_params.first_x - margin_x, plot_params.first_y - margin_y),
					(plot_params.last_x + margin_x, plot_params.last_y + margin_y),
					(prev_p.x, prev_p.y),
					(curr_p.x, curr_p.y),
				);
				// // TODO this is bad, we should do segment intersection instead
				// let on_screen = (prev_p.x >= plot_params.first_x - margin_x
				// 	&& prev_p.x <= plot_params.last_x + margin_x
				// 	&& prev_p.y >= plot_params.first_y - margin_y
				// 	&& prev_p.y <= plot_params.last_y + margin_y)
				// 	|| (curr_p.x >= plot_params.first_x - margin_x
				// 		&& curr_p.x <= plot_params.last_x + margin_x
				// 		&& curr_p.y >= plot_params.first_y - margin_y
				// 		&& curr_p.y <= plot_params.last_y + margin_y);

				cur_angle = Some(pseudoangle(dx, dy));
				if on_screen {
					let subdivisions_x =
						if a_dx > max_segment_x { (a_dx / max_segment_x).ceil() as usize } else { 1 };
					let subdivisions_y =
						if a_dy > max_segment_y { (a_dy / max_segment_y).ceil() as usize } else { 1 };
					let mut base_subdivisions = subdivisions_x.max(subdivisions_y);
					// max_base_subdivisions = max_base_subdivisions.max(base_subdivisions / 10);
					#[allow(clippy::integer_division)]
					let max_subdivisions = max_subdivisions.max(base_subdivisions / 10).min(plot_params.resolution / 5);

					// Adjust by curvature
					let adjusted_curvature = if let Some(prev_angle) = prev_angle {
						let curvature_factor = calculate_curvature_factor(prev_angle, cur_angle.unwrap());

						// 					let segment_length = (dx.powi(2) + dy.powi(2)).sqrt();
						// 					let length_scale = (max_segment_x.max(max_segment_y) /
						// segment_length).clamp(0.01, 1.0);

						// 					let adjusted_curvature = curvature_factor * length_scale.sqrt();
						// if debug {
						// 	println!("base subdivisions {subdivisions} curvature {curvature_factor}");
						// }
						curvature_factor

						// curvature_factors.push(adjusted_curvature);
					} else {
						// since we dont know how to calculate curvature, just use a conservative one
						8.0
					};
					if debug {
						println!("base subdivisions {base_subdivisions}");
					}
					base_subdivisions = (base_subdivisions as f64 * adjusted_curvature) as usize;
					if debug {
						println!("adjusted subdivisions {base_subdivisions}");
					}

					base_subdivisions = base_subdivisions.clamp(1, max_subdivisions);
					if debug {
						println!("clamped subdivisions {base_subdivisions} ");
					}

					if base_subdivisions > 1 {
						// Need to subdivide this segment
						let t_step = (curr_arg - prev_arg) / base_subdivisions as f64;
						for j in 1..base_subdivisions {
							let t = prev_arg + t_step * j as f64;
							let curr_p = eval_point_at(t).map_err(|e| (id, e))?;
							if debug {
								println!("subdivission {j} curr_p {curr_p:?}");
							}
							// Ok(Some((_, p))) => pp_buffer.push(p),
							// Ok(None) => {
							// 	// NaN in the middle
							// 	add_line(mem::take(&mut pp_buffer), true);
							// },
							// Err(e) => return Err((id, e)),
							// };

							if let Some((curr_arg, curr_p)) = curr_p {
								// Check for discontinuity
								if is_float2 {
									// TODO: this is BOTH wrong and bad.
									// We detect the discontinuity, but the _left and _right bisection results
									// are meaningless points (either (arg, y) or (arg,x), while we
									// need (x,y)) We either need to have custom disconinuity detector
									// for Float2 case, or at least dont do useless bisection inside
									// `detect(..)`
									if let Some((_left, _right)) = discontinuity_detector
										.detect(curr_arg, curr_p.y, |arg| {
											eval_point_at(arg).ok().and_then(|v| v.map(|p| p.1.y))
										})
										.or_else(|| {
											discontinuity_detector2.detect(curr_arg, curr_p.x, |arg| {
												eval_point_at(arg).ok().and_then(|v| v.map(|p| p.1.x))
											})
										}) {
										add_line(mem::take(&mut pp_buffer), true);
										add_point(&mut pp_buffer, curr_p, plot_params, &mut add_line);

										// prev_arg = curr_arg;
										// prev_angle = None;
										// prev_point = curr_result.map(|(_, v)| v);
										continue;
									}
								} else {
									if let Some((left, right)) = match TY {
										SimpleFunctionType::X => {
											discontinuity_detector.detect(curr_p.x, curr_p.y, |arg| {
												eval_point_at(arg).ok().and_then(|v| v.map(|p| p.1.y))
											})
										},
										SimpleFunctionType::Y => {
											discontinuity_detector.detect(curr_p.y, curr_p.x, |arg| {
												eval_point_at(arg).ok().and_then(|v| v.map(|p| p.1.x))
											})
										},
									} {
										match TY {
											SimpleFunctionType::X => {
												add_point(
													&mut pp_buffer,
													PlotPoint::new(left.0, left.1),
													plot_params,
													&mut add_line,
												);
												add_line(mem::take(&mut pp_buffer), true);
												add_point(
													&mut pp_buffer,
													PlotPoint::new(right.0, right.1),
													plot_params,
													&mut add_line,
												);
											},
											SimpleFunctionType::Y => {
												add_point(
													&mut pp_buffer,
													PlotPoint::new(left.1, left.0),
													plot_params,
													&mut add_line,
												);
												add_line(mem::take(&mut pp_buffer), true);
												add_point(
													&mut pp_buffer,
													PlotPoint::new(right.1, right.0),
													plot_params,
													&mut add_line,
												);
											},
										}
										// prev_arg = curr_arg;
										// prev_angle = None;
										// prev_point = curr_result.map(|(_, v)| v);
										continue;
									}
								}
							}
							if let Some((_, p)) = curr_p {
								add_point(&mut pp_buffer, p, plot_params, &mut add_line);
							} else {
								// NaN in the middle
								add_line(mem::take(&mut pp_buffer), true);
							};

							// } else {
							// }
						}
					}
					// num_splits.push(subdivisions);
				} else {
					// num_splits.push(1);
				}
				add_point(&mut pp_buffer, curr_p, plot_params, &mut add_line);
			},
			(Some(_), None) => {
				// nan
				add_line(mem::take(&mut pp_buffer), true);
			},
			(None, Some((_, p))) => {
				// coming back from NaN
				add_point(&mut pp_buffer, p, plot_params, &mut add_line);
			},
			(None, None) => {
				// Still NaN, nothing to do
			},
		}

		prev_angle = cur_angle;
		prev_arg = curr_arg;
		prev_point = curr_result.map(|(_, p)| p);
	}
	// println!(
	// 	"max_base_subdivisions {max_base_subdivisions} num_on_screen {num_on_screen} num_off_screen \
	// 	 {num_off_screen}"
	// );

	// if num_splits.len() > 0 {
	// num_splits.sort_unstable();
	// let avg_num_splits = num_splits.iter().sum::<usize>() as f64 / num_splits.len() as f64;
	// let total_num_splits = num_splits.iter().sum::<usize>();
	// println!(
	// 	"Average number of splits: {avg_num_splits} median {} max {} total {}",
	// 	num_splits[num_splits.len() / 2],
	// 	num_splits.last().unwrap(),
	// 	total_num_splits
	// );
	// }

	// if curvature_factors.len() > 0 {
	// 	curvature_factors.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
	// 	let avg_curvature_factor =
	// 		curvature_factors.iter().sum::<f64>() as f64 / curvature_factors.len() as f64;
	// 	let total_curvature_factor = curvature_factors.iter().sum::<f64>();
	// 	println!(
	// 		"Average curvature factor: {avg_curvature_factor} min {} median {} max {} total {}",
	// 		curvature_factors[0],
	// 		curvature_factors[curvature_factors.len() / 2],
	// 		curvature_factors.last().unwrap(),
	// 		total_curvature_factor
	// 	);
	// }
	add_line(pp_buffer, false);
	Ok(())
}
#[allow(clippy::too_many_arguments)]
fn draw_implicit<T: EvalexprFloat>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, plot_params: &PlotParams, id: u64,
	resolution: usize, equation_type: EquationType, color: Color32,
	call: impl Fn(&mut RefMut<Stack<T>>, f64, f64) -> Result<Value<T>, EvalexprError<T>> + Sync,
	mut add_line: impl FnMut(Vec<PlotPoint>), mut add_mesh: impl FnMut(MeshBuilder),
) -> Result<(), (u64, String)> {
	let mins = (plot_params.first_x, plot_params.first_y);
	let maxs = (plot_params.last_x, plot_params.last_y);

	let mut draw_lines = true;
	let mut draw_fill = None;
	match equation_type {
		EquationType::Equality => {},
		EquationType::LessThan => {
			draw_lines = false;
			draw_fill = Some(marching_squares::MarchingSquaresFill::Negative);
		},
		EquationType::LessThanOrEqual => {
			draw_fill = Some(marching_squares::MarchingSquaresFill::Negative);
		},
		EquationType::GreaterThan => {
			draw_lines = false;
			draw_fill = Some(marching_squares::MarchingSquaresFill::Positive);
		},
		EquationType::GreaterThanOrEqual => {
			draw_fill = Some(marching_squares::MarchingSquaresFill::Positive);
		},
		EquationType::None => {
			return Ok(());
		},
	}

	let bounds_diag = ((maxs.0 - mins.0).powi(2) + (maxs.1 - mins.1).powi(2)).sqrt();

	//146_327
	let eps = (T::EPSILON * bounds_diag * 100.0).max(T::EPSILON);
	//: 146_327

	let params = marching_squares::MarchingSquaresParams {
		resolution,
		bounds_min: mins,
		bounds_max: maxs,
		draw_lines,
		draw_fill,
		fill_color: color,
		eps,
	};
	let result = marching_squares::marching_squares(
		params,
		|cc: &mut RefMut<Stack<T>>, x, y| {
			let value = call(cc, x, y);
			// let value = match TY {
			// 	SimpleFunctionType::X => func.call(&mut *cc, ctx, &[f64_to_value::<T>(x)]),
			// 	SimpleFunctionType::Y => func.call(&mut *cc, ctx, &[f64_to_value::<T>(y)]),
			// };
			match value {
				Ok(v) => match v {
					Value::Float(v) => Ok(v.to_f64()),
					Value::Empty => Ok(f64::NAN),
					Value::Tuple(_) | Value::Float2(_, _) => {
						Err("Implicit function must return a single value.".to_string())
					},
					Value::Boolean(_) => Err("Implicit function must return a number.".to_string()),
				},
				Err(EvalexprError::ExpectedFloat { actual }) => {
					if actual == Value::Empty {
						Ok(f64::NAN)
					} else {
						Err(EvalexprError::ExpectedFloat { actual }.to_string())
					}
				},
				Err(EvalexprError::WrongTypeCombination { .. }) => {
					// TODO : this isnt good
					Ok(f64::NAN)
				},

				Err(e) => Err(e.to_string()),
			}
		},
		|| {
			let tl_context = thread_local_get(tl_context);
			// tl_context.stack_overflow_guard.set(0);
			tl_context.stack.borrow_mut()
		},
		&thread_local_get(tl_context).marching_squares_cache,
	);
	let result = match result {
		Ok(result) => result,
		Err(e) => return Err((id, e)),
	};
	// let total_line = result.iter().map(|r| r.lines.len()).sum::<usize>();
	// println!("total_lines: {total_line}");
	for result in result {
		for line in result.lines {
			add_line(line);
		}
		add_mesh(result.mesh);
	}

	Ok(())
}

pub fn gpu_position_from_point(plot_trans: &PlotTransform, invert_axes: [bool; 2], point: &PlotPoint) -> Pos2 {
	let x = remap(
		point.x,
		plot_trans.bounds().min()[0]..=plot_trans.bounds().max()[0],
		if invert_axes[0] { 1.0..=-1.0 } else { -1.0..=1.0 },
	) as f32;

	let y = remap(
		point.y,
		plot_trans.bounds().min()[1]..=plot_trans.bounds().max()[1],
		if invert_axes[1] { 1.0..=-1.0 } else { -1.0..=1.0 },
	) as f32;
	pos2(x, y)
}
