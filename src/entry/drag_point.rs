use eframe::egui::Id;
use egui_plot::PlotResponse;
use evalexpr::{EvalexprFloat, FlatNode, HashMapContext, IStr, Stack};

use crate::draw_buffer::{self, PointInteractionType};
use crate::entry::{DragPoint, Entry, EntryType, f64_to_float, f64_to_value};
use crate::math::{minimize, solve_secant};

pub struct DragPointResult {
	pub x:    f64,
	pub y:    f64,
	pub name: String,
}
pub fn point_dragging<T: EvalexprFloat>(
	entries: &mut [Entry<T>], ctx: &mut evalexpr::HashMapContext<T>, plot_res: &PlotResponse<()>,
	dragging_point_i: &mut Option<draw_buffer::PointInteraction>,
	hovered_point: Option<&(bool, draw_buffer::PointInteraction)>, eps: f64,
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
	let EntryType::Points { points, .. } = &mut entry.ty else {
		return None;
	};

	let point = &mut points[i.1 as usize];
	let point_x = dragging_point_i.x;
	let point_y = dragging_point_i.y;

	if let Some((x, y)) = point.val {
		result = Some(DragPointResult { x: x.to_f64(), y: y.to_f64(), name: entry.name.clone() });
	}

	let (Some(drag_point_type), Some(screen_pos)) = (point.drag_point.clone(), plot_res.response.hover_pos())
	else {
		return result;
	};

	let pos = plot_res.transform.value_from_position(screen_pos);
	match drag_point_type {
		DragPoint::BothCoordLiterals => {
			point.x.text = format!("{}", f64_to_float::<T>(pos.x));
			point.y.text = format!("{}", f64_to_float::<T>(pos.y));
		},
		DragPoint::XLiteral => {
			point.x.text = format!("{}", f64_to_float::<T>(pos.x));
		},
		DragPoint::YLiteral => {
			point.y.text = format!("{}", f64_to_float::<T>(pos.y));
		},
		DragPoint::XLiteralYConstant(y_const) => {
			point.x.text = format!("{}", f64_to_float::<T>(pos.x));
			let y_node = point.y.node.clone();
			drag(entries, ctx, y_const, y_node, point_y, pos.y, eps);
		},
		DragPoint::YLiteralXConstant(x_const) => {
			point.y.text = format!("{}", f64_to_float::<T>(pos.y));
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
			let new_value = minimize(value.to_f64(), (pos.x, pos.y), f32::EPSILON as f64, |x| {
				ctx.set_value(x_const, f64_to_value::<T>(x)).unwrap();
				(
					x_node.eval_float_with_context(&mut stack, ctx).unwrap().to_f64(),
					y_node.eval_float_with_context(&mut stack, ctx).unwrap().to_f64(),
				)
			});
			*value = f64_to_float::<T>(new_value);
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

	*value = f64_to_float::<T>(new_value);
}
fn get_entry_mut_by_id<T: EvalexprFloat>(entries: &mut [Entry<T>], id: Id) -> Option<&mut Entry<T>> {
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
