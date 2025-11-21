use alloc::sync::Arc;
use core::cell::RefMut;
use core::mem;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::marker::ConstParamTy;

use eframe::egui::{self, Color32, Id, RichText, Stroke};
use egui_plot::{Line, PlotPoint, PlotPoints, Points, Polygon, Text};
use evalexpr::{EvalexprError, EvalexprFloat, ExpressionFunction, FlatNode, HashMapContext, Stack, Value};
use thread_local::ThreadLocal;

use crate::draw_buffer::{
	DrawLine, DrawPoint, DrawPolygonGroup, DrawText, OtherPointType, PointInteraction, PointInteractionType
};
use crate::entry::{Entry, EntryType, f64_to_float, f64_to_value};
use crate::math::{DiscontinuityDetector, zoom_in_x_on_nan_boundary};
use crate::{ThreadLocalContext, marching_squares, thread_local_get};

pub struct PlotParams {
	pub eps:         f64,
	pub first_x:     f64,
	pub last_x:      f64,
	pub first_y:     f64,
	pub last_y:      f64,
	pub step_size:   f64,
	pub step_size_y: f64,
	pub resolution:  usize,
}
#[allow(clippy::too_many_arguments)]
#[allow(clippy::panic_in_result_fn)]
pub fn entry_create_plot_elements<T: EvalexprFloat>(
	entry: &mut Entry<T>, id: Id, sorting_idx: u32, selected_id: Option<Id>,
	ctx: &evalexpr::HashMapContext<T>, plot_params: &PlotParams,
	draw_buffer: &ThreadLocal<crate::DrawBufferRC>, tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>,
) -> Result<(), Vec<(u64, String)>> {
	// thread_local_get(tl_context).stack_overflow_guard.set(0);

	let visible = entry.visible;
	if !visible && !matches!(entry.ty, EntryType::Folder { .. }) {
		return Ok(());
	}
	let color = entry.color();
	// println!("step_size: {deriv_step_x}");

	let draw_buffer_c = thread_local_get(draw_buffer);
	match &mut entry.ty {
		EntryType::Folder { entries } => {
			let mut errors = std::sync::Mutex::new(Vec::new());
			entries.par_iter_mut().enumerate().for_each(|(ei, entry)| {
				let eid = Id::new(entry.id);
				if let Err(e) = entry_create_plot_elements(
					entry,
					eid,
					sorting_idx + ei as u32,
					selected_id,
					ctx,
					plot_params,
					draw_buffer,
					tl_context,
				) {
					errors.lock().unwrap().extend(e);
				}
			});
			if errors.get_mut().unwrap().is_empty() {
				return Ok(());
			}
			return Err(errors.into_inner().unwrap());
		},
		EntryType::Constant { .. } => {},
		EntryType::Label { x, y, size, underline, .. } => {
			let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
			let mut stack = thread_local_get(tl_context).stack.borrow_mut();

			match eval_point(&mut stack, ctx, x.node.as_ref(), y.node.as_ref()) {
				Ok(Some((x, y))) => {
					let size = if let Some(size) = &size.node {
						match size.eval_float_with_context(&mut stack, ctx) {
							Ok(size) => size.to_f64() as f32,
							Err(e) => {
								return Err(vec![(entry.id, e.to_string())]);
							},
						}
					} else {
						12.0
					};
					let mut label_text = RichText::new(entry.name.clone()).size(size);
					if *underline {
						label_text = label_text.underline()
					}

					let text = Text::new(entry.name.clone(), PlotPoint { x, y }, label_text).color(color);

					draw_buffer.texts.push(DrawText::new(sorting_idx, text));
				},
				Err(e) => {
					return Err(vec![(entry.id, e)]);
				},
				_ => {},
			}
		},
		EntryType::Points { points, style } => {
			let mut stack = thread_local_get(tl_context).stack.borrow_mut();
			let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
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
			let points_len = points.len();
			for (i, p) in points.iter_mut().enumerate() {
				match eval_point(&mut stack, ctx, p.x.node.as_ref(), p.y.node.as_ref()) {
					Ok(Some((x, y))) => {
						p.val = Some((f64_to_float::<T>(x), f64_to_float::<T>(y)));
						let point_id = id.with(i);
						let selected = selected_id == Some(id);
						let radius = if selected { 6.5 } else { 4.5 };
						let radius_outer = if selected { 12.5 } else { 7.5 };

						if style.show_points || p.drag_point.is_some() {
							draw_buffer.points.push(DrawPoint::new(
								sorting_idx,
								i as u32,
								PointInteraction {
									x,
									y,
									radius,
									ty: PointInteractionType::Other(OtherPointType::Point),
								},
								Points::new(entry.name.clone(), [x, y]).color(color).radius(radius),
							));
							if p.drag_point.is_some() {
								let selectable_point = PointInteraction {
									ty: PointInteractionType::Draggable { i: (id, i as u32) },
									x,
									y,
									radius: radius_outer,
								};
								draw_buffer.points.push(DrawPoint::new(
									sorting_idx,
									i as u32,
									selectable_point,
									Points::new(entry.name.clone(), [x, y])
										.id(point_id)
										.color(color_outer)
										.radius(radius_outer),
								));
							}
						}
						if let Some(label_config) = &style.label_config
							&& i == points_len - 1
							&& !entry.name.trim().is_empty()
						{
							let size = label_config.size.size();
							let mut label = RichText::new(entry.name.clone()).size(size);
							if label_config.italic {
								label = label.italics();
							}

							let dir = label_config.pos.dir();
							let text = Text::new(
								entry.name.clone(),
								PlotPoint {
									x: x + (dir.x * size * arrow_scale.x) as f64,
									y: y + (dir.y * size * arrow_scale.y) as f64,
								},
								label,
							)
							.color(color);

							draw_buffer.texts.push(DrawText::new(sorting_idx, text));
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
					},
					Err(e) => {
						return Err(vec![(entry.id, e)]);
					},
					_ => {},
				}
			}

			// let width = if selected { 3.5 } else { 1.0 };
			let width = 1.0;

			if line_buffer.len() > 1 && style.show_lines {
				let line = Line::new(entry.name.clone(), line_buffer)
					.color(color)
					.id(id)
					.width(style.line_style.line_width)
					.style(style.line_style.egui_line_style());
				draw_buffer.lines.push(DrawLine::new(sorting_idx, id, width, line));
				if !arrow_buffer.is_empty() {
					draw_buffer.polygons.push(DrawPolygonGroup::new(sorting_idx, arrow_buffer));
				}
				// plot_ui.line(line);
			}
		},
		EntryType::Function {
			func,
			can_be_drawn,
			identifier,
			parametric,
			range_start,
			range_end,
			style,
			implicit_resolution,
			..
		} => {
			if !*can_be_drawn {
				return Ok(());
			}

			let display_name = if identifier.is_empty() {
				format!("f({}): {}", func.args_to_string(), func.text.trim())
			} else {
				format!("{}({}): {}", identifier, func.args_to_string(), func.text.trim())
			};

			let selected = selected_id == Some(id);
			let width = if selected { style.line_width + 2.5 } else { style.line_width };

			let mut add_line = |line: Vec<PlotPoint>| {
				if line.is_empty() {
					return;
				}
				let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
				draw_buffer.lines.push(DrawLine::new(
					sorting_idx,
					id,
					width,
					Line::new(&display_name, PlotPoints::Owned(line))
						.id(id)
						.width(width)
						.style(style.egui_line_style())
						.color(color),
				));
			};

			if let Some(expr_func) = &func.expr_function {
				if func.args.is_empty() {
					let mut stack = thread_local_get(tl_context).stack.borrow_mut();
					// horizontal line
					let value = expr_func
						.call(&mut stack, ctx, &[])
						.and_then(|v| v.as_float())
						.map_err(|e| vec![(entry.id, e.to_string())])?;
					add_line(vec![
						PlotPoint::new(plot_params.first_x, value.to_f64()),
						PlotPoint::new(plot_params.last_x, value.to_f64()),
					]);
				} else if func.args.len() == 1 {
					// starndard X or Y function
					if *parametric {
						if func.args[0].to_str() == "y" {
							draw_parametric_function::<T, { SimpleFunctionType::Y }>(
								tl_context,
								ctx,
								plot_params,
								entry.id,
								expr_func,
								range_start.node.as_ref(),
								range_end.node.as_ref(),
								&mut add_line,
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
								&mut add_line,
							)?;
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
							// draw_y_function(tl_context, ctx, plot_params, entry.id, expr_func, &mut
							// add_line)?;
						}
					}
				} else if func.args.len() == 2 {
					if func.args[0].to_str() == "x" && func.args[1].to_str() == "y" {
						draw_xy_function(
							tl_context, ctx, plot_params, entry.id, expr_func, *implicit_resolution,
							&mut add_line,
						)?;
					}
				}
			}
		},
	}
	Ok(())
}

fn eval_point<T: EvalexprFloat>(
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
) -> Result<(), Vec<(u64, String)>> {
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
		let value = constant.as_float().map_err(|e| vec![(id, e.to_string())])?;
		pp_buffer.push(PlotPoint::new(plot_params.first_x, value.to_f64()));
		pp_buffer.push(PlotPoint::new(plot_params.last_x, value.to_f64()));
	} else {
		// let mut prev_y = None;
		let mut discontinuity_detector = DiscontinuityDetector::new(step_size, plot_params.eps);

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

					if !on_nan_boundary
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
								pp_buffer.push(PlotPoint::new(right.0, right.1));
							},
							SimpleFunctionType::Y => {
								pp_buffer.push(PlotPoint::new(left.1, left.0));
								add_line(mem::take(&mut pp_buffer));
								pp_buffer.push(PlotPoint::new(right.1, right.0));
							},
						}
					} else {
						if !cur_val.is_nan() {
							match TY {
								SimpleFunctionType::X => {
									pp_buffer.push(PlotPoint::new(cur_arg, cur_val));
								},
								SimpleFunctionType::Y => {
									pp_buffer.push(PlotPoint::new(cur_val, cur_arg));
								},
							};
						}
					}

					prev_sampling_point = Some((sampling_arg, val));
				},
				Ok(Value::Empty) => {
					add_line(mem::take(&mut pp_buffer));
				},
				Ok(Value::Tuple(_) | Value::Float2(_, _)) => {
					return Err(vec![(
						id,
						"Non-parametric function must return a single number.".to_string(),
					)]);
				},
				Ok(Value::Boolean(_)) => {
					return Err(vec![(id, "Non-parametric function must return a number.".to_string())]);
				},
				Err(e) => {
					return Err(vec![(id, e.to_string())]);
				},
			}

			let prev_sampling_arg = sampling_arg;
			sampling_arg += step_size;
			if sampling_arg == prev_sampling_arg {
				break;
			}
		}
	}

	add_line(pp_buffer);

	Ok(())
}
#[allow(clippy::too_many_arguments)]
fn draw_parametric_function<T: EvalexprFloat, const TY: SimpleFunctionType>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, ctx: &HashMapContext<T>, plot_params: &PlotParams,
	id: u64, func: &ExpressionFunction<T>, range_start: Option<&FlatNode<T>>, range_end: Option<&FlatNode<T>>,
	mut add_line: impl FnMut(Vec<PlotPoint>),
) -> Result<(), Vec<(u64, String)>> {
	let mut stack = thread_local_get(tl_context).stack.borrow_mut();
	let mut pp_buffer = vec![];
	match eval_point(&mut stack, ctx, range_start, range_end) {
		Ok(Some((start, end))) => {
			if start > end {
				return Err(vec![(id, "Range start must be less than range end".to_string())]);
			}
			let range = end - start;
			let step = range / plot_params.resolution as f64;
			if start + step == end {
				return Ok(());
			}
			let mut discontinuity_detector = DiscontinuityDetector::new(step, plot_params.eps);
			for i in 0..(plot_params.resolution + 1) {
				let arg = start + step * i as f64;
				match func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)]) {
					Ok(Value::Float(value)) => {
						if let Some((left, right)) =
							discontinuity_detector.detect(arg, value.to_f64(), |arg| {
								func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)])
									.and_then(|v| v.as_float())
									.map(|v| v.to_f64())
									.ok()
							}) {
							// pp_buffer.push(PlotPoint::new(left.0, left.0));
							// add_line(mem::take(&mut pp_buffer));
							// pp_buffer.push(PlotPoint::new(right.0, right.0));

							match TY {
								SimpleFunctionType::X => {
									pp_buffer.push(PlotPoint::new(left.0, left.1));
									add_line(mem::take(&mut pp_buffer));
									pp_buffer.push(PlotPoint::new(right.0, right.1));
								},
								SimpleFunctionType::Y => {
									pp_buffer.push(PlotPoint::new(left.1, left.0));
									add_line(mem::take(&mut pp_buffer));
									pp_buffer.push(PlotPoint::new(right.1, right.0));
								},
							}
						} else {
							if !value.is_nan() {
								match TY {
									SimpleFunctionType::X => {
										pp_buffer.push(PlotPoint::new(arg, value.to_f64()));
									},
									SimpleFunctionType::Y => {
										pp_buffer.push(PlotPoint::new(value.to_f64(), arg));
									},
								};
							}
						}
					},
					Ok(Value::Empty) => {},
					Ok(Value::Float2(x, y)) => {
						// TODO: detect discontinuity for ranged thar teturn Float2
						if y.is_nan() {
							if !pp_buffer.is_empty() {
								add_line(mem::take(&mut pp_buffer));
							}
						} else {
							pp_buffer.push(PlotPoint::new(x.to_f64(), y.to_f64()));
						}
					},
					Ok(_) => {
						// println!("ranged function must return 1 or 2 float values");
						return Err(vec![(id, "Ranged function must return 1 or 2 float values".to_string())]);
					},
					Err(e) => {
						// println!("error {e}");
						return Err(vec![(id, e.to_string())]);
					},
				}
			}
			add_line(pp_buffer);
		},
		Err(e) => {
			return Err(vec![(id, e)]);
		},
		_ => {},
	}

	Ok(())
}
fn draw_xy_function<T: EvalexprFloat>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, ctx: &HashMapContext<T>, plot_params: &PlotParams,
	id: u64, func: &ExpressionFunction<T>, resolution: usize, mut add_line: impl FnMut(Vec<PlotPoint>),
) -> Result<(), Vec<(u64, String)>> {
	let mins = (plot_params.first_x, plot_params.first_y);
	let maxs = (plot_params.last_x, plot_params.last_y);

	for (_, lines) in marching_squares::marching_squares(
		|cc: &mut RefMut<Stack<T>>, x, y| match func.call(
			&mut *cc,
			ctx,
			&[f64_to_value::<T>(x), f64_to_value::<T>(y)],
		) {
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
			Err(e) => Err(e.to_string()),
		},
		mins,
		maxs,
		resolution,
		|| {
			let tl_context = thread_local_get(tl_context);
			// tl_context.stack_overflow_guard.set(0);
			tl_context.stack.borrow_mut()
		},
		&thread_local_get(tl_context).marching_squares_cache,
	)
	.map_err(|e| vec![(id, e)])?
	{
		for line in lines {
			add_line(line);
		}
	}

	Ok(())
}
