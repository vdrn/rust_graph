use eframe::egui::Id;
use egui_plot::PlotResponse;
use evalexpr::{EvalexprFloat, FlatNode, HashMapContext, IStr, Stack};

use crate::draw_buffer::{self, PointInteractionType};
use crate::entry::{DragPoint, Entry, EntryType, PlotParams, PointsType, f64_to_value};
use crate::math::{minimize, solve_secant};

pub struct DragPointResult {
	pub x:    f64,
	pub y:    f64,
	pub name: String,
}
pub fn point_dragging<T: EvalexprFloat>(
	entries: &mut [Entry<T>], ctx: &mut evalexpr::HashMapContext<T>, plot_res: &PlotResponse<()>,
	dragging_point_i: &mut Option<draw_buffer::PointInteraction>,
	hovered_point: Option<&(bool, draw_buffer::PointInteraction)>, plot_params: &PlotParams,
) -> Option<DragPointResult> {
	let mut result = None;

	if let Some((is_draggable, hovered_point)) = &hovered_point {
		if *is_draggable && plot_res.response.drag_started() {
			*dragging_point_i = Some(hovered_point.clone());
		}
	}
	if plot_res.response.drag_stopped() {
		*dragging_point_i = None;
	}
	let Some(dragging_point_i) = dragging_point_i else {
		return result;
	};
	let PointInteractionType::Draggable { i } = dragging_point_i.ty else {
		return result;
	};
	let Some(entry) = get_entry_mut_by_id(entries, i.0) else {
		return result;
	};
	let EntryType::Points { points_ty, .. } = &mut entry.ty else {
		return None;
	};
	let PointsType::Separate(points) = points_ty else {
		return None;
	};

	let point = &mut points[i.1 as usize];
	let point_x = dragging_point_i.x;
	let point_y = dragging_point_i.y;

	if let Some((x, y)) = point.val {
		result = Some(DragPointResult { x: x.to_f64(), y: y.to_f64(), name: entry.name.clone() });
	}

	let (Some(drag_point_type), Some(screen_pos)) =
		(point.drag.drag_point.clone(), plot_res.response.hover_pos())
	else {
		return result;
	};

	let scale_x = plot_params.last_x - plot_params.first_x;
	let scale_y = plot_params.last_y - plot_params.first_y;
	let min_scale = scale_x.min(scale_y);
	let eps = plot_params.eps * f64::max(min_scale, 100.0);
	let pos = plot_res.transform.value_from_position(screen_pos);
	match drag_point_type {
		DragPoint::BothCoordLiterals => {
			point.x.text = to_string_with_scale(T::from_f64(pos.x), scale_x);
			point.y.text = to_string_with_scale(T::from_f64(pos.y), scale_y);
		},
		DragPoint::XLiteral => {
			point.x.text = to_string_with_scale(T::from_f64(pos.x), scale_x);
		},
		DragPoint::YLiteral => {
			point.y.text = to_string_with_scale(T::from_f64(pos.y), scale_y);
		},
		DragPoint::XLiteralYConstant(y_const) => {
			point.x.text = to_string_with_scale(T::from_f64(pos.x), scale_x);
			let y_node = point.y.node.clone();
			drag(entries, ctx, y_const, y_node, point_y, pos.y, eps);
		},
		DragPoint::YLiteralXConstant(x_const) => {
			point.y.text = to_string_with_scale(T::from_f64(pos.y), scale_y);
			let x_node = point.x.node.clone();
			drag(entries, ctx, x_const, x_node, point_x, pos.x, eps);
		},
		DragPoint::BothCoordConstants(x_const, y_const) => {
			let x_node = point.x.node.clone();
			let y_node = point.y.node.clone();

			drag(entries, ctx, x_const, x_node, point_x, pos.x, eps);
			drag(entries, ctx, y_const, y_node, point_y, pos.y, eps);
		},
		DragPoint::XConstant(x_const) => {
			let x_node = point.x.node.clone();
			drag(entries, ctx, x_const, x_node, point_x, pos.x, eps);
		},
		DragPoint::YConstant(y_const) => {
			let y_node = point.y.node.clone();
			drag(entries, ctx, y_const, y_node, point_y, pos.y, eps);
		},
		DragPoint::SameConstantBothCoords(x_const) => {
			let x_node = point.x.node.clone();
			let y_node = point.y.node.clone();

			let (Some(x_node), Some(y_node)) = (x_node, y_node) else {
				return result;
			};
			let Some(value) = find_constant_value(entries, |e| e == x_const) else {
				return result;
			};

			let mut stack = Stack::<T>::new();
			let new_value = minimize(value.to_f64(), (pos.x, pos.y), eps , |x| {
				ctx.set_value(x_const, f64_to_value::<T>(x)).unwrap();
				Some((
					x_node.eval_float_with_context(&mut stack, ctx).ok()?.to_f64(),
					y_node.eval_float_with_context(&mut stack, ctx).ok()?.to_f64(),
				))
			});
			if let Some(new_value) = new_value {
				*value = T::from_f64(new_value);
			}
		},
	}

	result
}

fn drag<T: EvalexprFloat>(
	entries: &mut [Entry<T>], ctx: &mut HashMapContext<T>, name: IStr, node: Option<FlatNode<T>>, cur: f64,
	target: f64, eps: f64,
) {
	let Some(point_node) = node else {
		return;
	};
	let Some(value) = find_constant_value(entries, |e| e == name) else {
		return;
	};

	let mut stack = Stack::<T>::new();
	let Some(new_value) = solve_secant(value.to_f64(), cur, target, eps, |x| {
		ctx.set_value(name, f64_to_value::<T>(x)).unwrap();
		point_node.eval_float_with_context(&mut stack, ctx).unwrap().to_f64()
	}) else {
		return;
	};

	*value = T::from_f64(new_value);
}
pub fn get_entry_mut_by_id<T: EvalexprFloat>(entries: &mut [Entry<T>], id: Id) -> Option<&mut Entry<T>> {
	for entry in entries.iter_mut() {
		if Id::new(entry.id) == id {
			return Some(entry);
		} else if let EntryType::Folder { entries } = &mut entry.ty {
			for sub_entry in entries.iter_mut() {
				if Id::new(sub_entry.id) == id {
					return Some(sub_entry);
				}
			}
		}
	}
	None
}

fn find_constant_value<T: EvalexprFloat>(
	entries: &mut [Entry<T>], pos_cb: impl Fn(IStr) -> bool,
) -> Option<&mut T> {
	for entry in entries.iter_mut() {
		match &mut entry.ty {
			EntryType::Constant { value, istr_name, .. } => {
				if pos_cb(*istr_name) {
					return Some(value);
				}
			},
			EntryType::Folder { entries } => {
				for sub_entry in entries.iter_mut() {
					if let EntryType::Constant { value, istr_name, .. } = &mut sub_entry.ty {
						if pos_cb(*istr_name) {
							return Some(value);
						}
					}
				}
			},
			_ => {},
		}
	}
	None
}
pub fn decimal_places_for_scale<T: EvalexprFloat>(scale: f64) -> i32 {
	if scale <= 0.0 {
		return 2;
	}

	// order of magnitude
	let log_scale = scale.log10();

	let decimals = 4 - log_scale.ceil() as i32;

	decimals.clamp(-2, T::HUMAN_DISPLAY_SIG_DIGITS as i32)
}

pub fn round_to_decimals<T: EvalexprFloat>(value: T, decimals: i32) -> T {
	if decimals >= 0 {
		let multiplier = T::from_f64(10_f64.powi(decimals));
		(value * multiplier).round() / multiplier
	} else {
		// For negative decimals, round to nearest 10, 100, etc.
		let divisor = T::from_f64(10_f64.powi(-decimals));
		(value / divisor).round() * divisor
	}
}
pub fn to_string_with_scale<T: EvalexprFloat>(value: T, scale: f64) -> String {
	let prec = decimal_places_for_scale::<T>(scale);
	let value = round_to_decimals::<T>(value, prec);

	let prec_usize = prec.max(0) as usize;
	format!("{:.prec$}", value, prec = prec_usize)
}
