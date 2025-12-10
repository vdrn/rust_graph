use core::ops::RangeInclusive;

use eframe::egui::containers::menu::{MenuButton, MenuConfig, SubMenuButton};
use eframe::egui::text::CCursorRange;
use eframe::egui::text_edit::TextEditState;
use eframe::egui::{
	self, Align, Button, Color32, DragValue, Label, PopupCloseBehavior, RichText, Slider, TextEdit, TextWrapMode, Widget, vec2
};

use evalexpr::{EvalexprFloat, HashMapContext, Stack};

use crate::custom_rendering::fan_fill_renderer::FillRule;
use crate::entry::entry_processing::preprocess_ast;
use crate::entry::{
	COLORS, ConstantType, DragPoint, Entry, EntryType, EquationType, Expr, LabelConfig, LabelPosition, LabelSize, LineStyleConfig, LineStyleType, MAX_IMPLICIT_RESOLUTION, MIN_IMPLICIT_RESOLUTION, PointDrag, PointDragType, PointEntry, PointsType, ProcessedColors, RESERVED_NAMES
};
use crate::widgets::{duplicate_entry_btn, full_width_slider, remove_entry_btn};

pub struct EditEntryResult {
	pub needs_recompilation: bool,
	pub animating:           bool,
	pub needs_redraw:        bool,
	pub remove:              bool,
	pub duplicate:           bool,
	pub error:               Option<String>,
	pub parsed:              bool,
}
pub fn entry_ui<T: EvalexprFloat>(
	ui: &mut egui::Ui, ctx: &HashMapContext<T>,
  processed_colors: &ProcessedColors<T>,

  entry: &mut Entry<T>, clear_cache: bool,
) -> EditEntryResult {
	let mut result = EditEntryResult {
		needs_recompilation: false,
		animating:           false,
		remove:              false,
		needs_redraw:        false,
		duplicate:           false,
		parsed:              false,

		error: None,
	};

	let (text_col, fill_col) = if entry.active {
		(Color32::BLACK, entry.color())
	} else {
		(Color32::LIGHT_GRAY, egui::Color32::TRANSPARENT)
	};

	let prev_visible = entry.active;
	// name and visibility
	ui.horizontal(|ui| {
		if ui
			.add(
				Button::new(RichText::new(entry.entry_symbol().symbol(entry.active)).strong().monospace().color(text_col))
					.fill(fill_col)
					.corner_radius(10),
			)
			.on_hover_text(match &entry.ty {
				EntryType::Constant { .. } => {
					if entry.active {
						"Pause"
					} else {
						"Play"
					}
				},
				_ => {
					if entry.active {
						"Hide in Graph"
					} else {
						"Show in Graph"
					}
				},
			})
			.clicked()
		{
			entry.active = !entry.active;
			result.needs_redraw = true;
		}

		let name_was_ok = !RESERVED_NAMES.contains(&entry.name.trim());
		if ui
			.add(TextEdit::singleline(&mut entry.name).desired_width(30.0).clip_text(false).hint_text("name"))
			.changed()
		{
			result.needs_recompilation = true;
			result.parsed = true;
		}
		if RESERVED_NAMES.contains(&entry.name.trim()) {
			result.error = Some(format!("{} is reserved name.", entry.name));
			// return;
		} else if !name_was_ok {
			result.parsed = true;
		}
	});
	// entry_edit and controls

	ui.with_layout(egui::Layout::centered_and_justified(egui::Direction::RightToLeft), |ui| {
		// controls
		ui.horizontal(|ui| {
			ui.with_layout(egui::Layout::top_down(Align::RIGHT), |ui| {
				if remove_entry_btn(ui, entry.entry_symbol().name()) {
					result.remove = true;
					result.needs_recompilation = true;
				}
				if duplicate_entry_btn(ui, entry.entry_symbol().name()) {
					result.duplicate = true;
					result.needs_recompilation = true;
				}
			});
			ui.with_layout(egui::Layout::top_down(Align::RIGHT), |ui| {
				if entry_style(ui, entry) {
					result.needs_redraw = true;
				}
			});
		});
		// entry edit
		ui.horizontal(|ui| {
			entry_type_ui(ui, ctx,processed_colors, entry, clear_cache, prev_visible, &mut result);
		});
	});

	result
}

fn entry_style<T: EvalexprFloat>(ui: &mut egui::Ui, entry: &mut Entry<T>) -> bool {
	let mut changed = false;
	let mut style_button = MenuButton::new(RichText::new("🎨").color(Color32::BLACK))
		.config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside));
	style_button.button = style_button.button.fill(entry.color());
	style_button
		.ui(ui, |ui| {
			let mut color_button = SubMenuButton::new(RichText::new("Color").color(Color32::BLACK));
			color_button.button = color_button.button.fill(entry.color());

			color_button.ui(ui, |ui| {
				for i in 0..COLORS.len() {
					if ui
						.button(RichText::new("     ").background_color(COLORS[i]))
						.on_hover_text("Change Color")
						.clicked()
					{
						changed = true;
						entry.color = i;
					}
				}
			});
			match &mut entry.ty {
				EntryType::Function { style, parametric, parametric_fill, fill_rule, .. } => {
					ui.separator();

					changed |= line_style_config_ui(style, ui);
					ui.separator();
					if *parametric {
						changed |= ui
							.checkbox(parametric_fill, "Fill")
							.on_hover_text("Fill the inside area the curve.")
							.clicked();
						if *parametric_fill {
							changed |= fill_rule_btn_ui(ui, fill_rule);
						}
					}

					egui::Sides::new().show(
						ui,
						|_ui| {},
						|ui| {
							if ui.button("Close").clicked() {
								ui.close();
							}
							if ui.button("Reset").clicked() {
								*style = Default::default();
								changed = true;
								ui.close();
							}
						},
					);
				},
				EntryType::Points { style, .. } => {
					ui.separator();
					let mut show_label = style.label_config.is_some();
					if ui.checkbox(&mut show_label, "Show label").changed() {
						changed = true;
						if show_label {
							style.label_config = Some(LabelConfig::default());
						} else {
							style.label_config = None;
						}
					}
					if let Some(label_config) = &mut style.label_config {
						changed |= label_config_ui(label_config, ui);
					}
					ui.separator();
					changed |= ui.checkbox(&mut style.show_lines, "Show Lines").clicked();
					if style.show_lines {
						changed |= ui
							.checkbox(&mut style.connect_first_and_last, "Connect first and last point")
							.clicked();
						ui.label("Line Style:");
						changed |= ui.checkbox(&mut style.show_arrows, "Show Arrows").clicked();
						ui.separator();
						changed |= line_style_config_ui(&mut style.line_style, ui);
					} else {
						style.show_arrows = false;
					}
					ui.separator();
					changed |= ui
						.checkbox(&mut style.fill, "Fill")
						.on_hover_text("Fill the area between the points.")
						.clicked();
					if style.fill {
						changed |= fill_rule_btn_ui(ui, &mut style.fill_rule);
					}

					ui.separator();
					changed |= ui.checkbox(&mut style.show_points, "Show Not Draggable Points").clicked();

					egui::Sides::new().show(
						ui,
						|_ui| {},
						|ui| {
							if ui.button("Close").clicked() {
								ui.close();
							}
							if ui.button("Reset").clicked() {
								*style = Default::default();
								changed = true;
								ui.close();
							}
						},
					);
				},
				EntryType::Constant { .. } => {},
				EntryType::Folder { .. } => {},
				EntryType::Color(_) => {},
			}
		})
		.0
		.on_hover_text("Edit Apprearance");
	changed
}
fn entry_type_ui<T: EvalexprFloat>(
	ui: &mut egui::Ui, ctx: &HashMapContext<T>, processed_colors: &ProcessedColors<T>, entry: &mut Entry<T>,
	clear_cache: bool, prev_active: bool, result: &mut EditEntryResult,
) {
	match &mut entry.ty {
		EntryType::Folder { .. } => {
			// handled in outer scope
		},
		EntryType::Function { func, parametric, range_start, range_end, implicit_resolution, .. } => {
			ui.vertical(|ui| {
				match expr_ui(func, ui, "sin(x)", None, clear_cache, true) {
					Ok(changed) => {
						result.parsed |= changed;
						result.needs_recompilation |= changed;
					},
					Err(e) => {
						result.error = Some(format!("Parsing error: {e}"));
					},
				}

				ui.vertical(|ui| {
					if let Some(computed_const) = func.computed_const().cloned() {
						ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
							Label::new(computed_const.human_display(func.display_rational))
								.wrap_mode(TextWrapMode::Truncate)
								.selectable(true)
								.ui(ui);

							ui.label("=");
							if Button::new(if func.display_rational { "Q" } else { "F" })
								.ui(ui)
								.on_hover_text(if func.display_rational {
									"Display as Float"
								} else {
									"Display as Rational"
								})
								.clicked()
							{
								func.display_rational = !func.display_rational;
							}
						});
					}
					if func.args.len() == 1 && func.equation_type == EquationType::None {
						ui.horizontal_wrapped(|ui| {
							if ui.checkbox(parametric, "Parametric").changed() {
								result.needs_redraw = true;
							}
							if *parametric && func.args.len() == 1 {
								ui.label("Start:");
								match expr_ui(range_start, ui, "", Some(30.0), clear_cache, false) {
									Ok(changed) => {
										result.needs_recompilation |= changed;
										result.parsed |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}
								ui.label("End:");
								match expr_ui(range_end, ui, "", Some(30.0), clear_cache, false) {
									Ok(changed) => {
										result.needs_recompilation |= changed;
										result.parsed |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}
							}
						});
						ui.label("Parametric fns can return 1 or 2 values: f(x)->y  or f(t)->(x,y)");
					} else if func.equation_type != EquationType::None {
						*parametric = false;
					}
					if func.args.len() == 2 && func.args[0].to_str() == "x" && func.args[1].to_str() == "y" {
						ui.horizontal(|ui| {
							if Slider::new(
								implicit_resolution,
								MIN_IMPLICIT_RESOLUTION..=MAX_IMPLICIT_RESOLUTION,
							)
							.text("Implicit Resolution")
							.ui(ui)
							.changed()
							{
								result.needs_redraw = true;
							}
						});
					}
				});
			});
		},
		EntryType::Points { points_ty, style, .. } => {
			let mut remove_point = None;
			ui.vertical(|ui| {
				match points_ty {
					PointsType::Separate(points) => {
						for (pi, point) in points.iter_mut().enumerate() {
							ui.horizontal(|ui| {
								match expr_ui(&mut point.x, ui, "point_x", Some(80.0), clear_cache, false) {
									Ok(changed) => {
										result.parsed |= changed;
										result.needs_recompilation |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}
								match expr_ui(&mut point.y, ui, "point_y", Some(80.0), clear_cache, false) {
									Ok(changed) => {
										result.parsed |= changed;
										result.needs_recompilation |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}
								let mut drag_type_changed = false;
								if !point.drag.both_drag_dirs_available
									&& point.drag.drag_type == PointDragType::Both
								{
									point.drag.both_drag_dirs_available = false;
									drag_type_changed = true;
								}

								let drag_menu_text = match &point.drag.drag_point {
									Some(d) => match d {
										DragPoint::BothCoordLiterals => {
											format!("{}(x,y)", point.drag.drag_type.symbol())
										},
										DragPoint::XLiteral => {
											format!("{}(x,_)", point.drag.drag_type.symbol())
										},
										DragPoint::YLiteral => {
											format!("{}(_,y)", point.drag.drag_type.symbol())
										},
										DragPoint::XConstant(x) => {
											format!("{}({},_)", point.drag.drag_type.symbol(), x)
										},
										DragPoint::YConstant(y) => {
											format!("{}(_, {})", point.drag.drag_type.symbol(), y)
										},
										DragPoint::XLiteralYConstant(y) => {
											format!("{}(x, {})", point.drag.drag_type.symbol(), y)
										},
										DragPoint::YLiteralXConstant(x) => {
											format!("{}({}, y)", point.drag.drag_type.symbol(), x)
										},
										DragPoint::BothCoordConstants(x, y) => {
											format!("{}({}, {})", point.drag.drag_type.symbol(), x, y,)
										},
										DragPoint::SameConstantBothCoords(x) => {
											format!("{}({})", point.drag.drag_type.symbol(), x)
										},
									},
									None => point.drag.drag_type.symbol().to_string(),
								};
								ui.menu_button(drag_menu_text, |ui| {
									drag_type_changed |= ui
										.selectable_value(
											&mut point.drag.drag_type,
											PointDragType::NoDrag,
											PointDragType::NoDrag.name(),
										)
										.changed();
									if point.drag.both_drag_dirs_available {
										drag_type_changed |= ui
											.selectable_value(
												&mut point.drag.drag_type,
												PointDragType::Both,
												PointDragType::Both.name(),
											)
											.changed();
									}
									drag_type_changed |= ui
										.selectable_value(
											&mut point.drag.drag_type,
											PointDragType::X,
											PointDragType::X.name(),
										)
										.changed();
									drag_type_changed |= ui
										.selectable_value(
											&mut point.drag.drag_type,
											PointDragType::Y,
											PointDragType::Y.name(),
										)
										.changed();
								})
								.response
								.on_hover_text("Choose dragging behavior.");
								if drag_type_changed {
									result.parsed = true;
									result.needs_recompilation = true;
								}

								if ui.button("❌").on_hover_text("Remove Point").clicked() {
									remove_point = Some(pi);
								}
							});
						}
						if let Some(pi) = remove_point {
							points.remove(pi);
							result.needs_recompilation = true;
						}
					},
					PointsType::SingleExpr { expr, .. } => {
						ui.horizontal(|ui| match expr_ui(expr, ui, "Points", None, clear_cache, false) {
							Ok(changed) => {
								result.parsed |= changed;
								result.needs_recompilation |= changed;
							},
							Err(e) => {
								result.error = Some(format!("Parsing error: {e}"));
							},
						});
					},
				}

				ui.horizontal(|ui| {
					let mut is_single_expr = true;
					if let PointsType::Separate(points) = points_ty {
						is_single_expr = false;
						if ui.button("Add Point").clicked() {
							points.push(PointEntry::default());
						}
					}
					if ui.checkbox(&mut is_single_expr, "Single Expression").clicked() {
						result.needs_recompilation = true;
						match points_ty {
							PointsType::Separate(points) => {
								let mut expr_from_points = String::new();
								for (i, p) in points.iter().enumerate() {
									if i > 0 {
										expr_from_points.push_str(", ");
									}
									expr_from_points.push('(');
									expr_from_points.push_str(&p.x.text);
									expr_from_points.push_str(", ");
									expr_from_points.push_str(&p.y.text);
									expr_from_points.push(')');
								}
								*points_ty = PointsType::SingleExpr {
									expr: Expr::from_text(&expr_from_points),
									val:  Vec::new(),
								};
							},
							PointsType::SingleExpr { expr: _, val } => {
								let value_as_points = val
									.iter()
									.map(|v| PointEntry {
										x:    Expr::from_text(&format!("{}", v.0)),
										y:    Expr::from_text(&format!("{}", v.1)),
										drag: PointDrag {
											drag_point:               None,
											drag_type:                PointDragType::NoDrag,
											both_drag_dirs_available: true,
										},
										val:  Some((v.0, v.1)),
									})
									.collect();
								*points_ty = PointsType::Separate(value_as_points);
							},
						}
					}

					let mut show_label = style.label_config.is_some();
					if ui.checkbox(&mut show_label, "Show label").changed() {
						result.needs_redraw = true;
						if show_label {
							style.label_config = Some(LabelConfig::default());
						} else {
							style.label_config = None;
						}
					}
					if let Some(label_config) = &mut style.label_config {
						result.needs_redraw |= TextEdit::singleline(&mut label_config.text).ui(ui).changed();
					}
				});
			});
		},

		EntryType::Constant { value, step, ty, range_start, range_end, .. } => {
			let original_value = *value;
			let step_f = T::from_f64(*step);

			let mut range_error = None;
			let mut range = match ty.range(ctx, range_start, range_end) {
				Ok(range) => range,
				Err(err) => {
					range_error = Some(err);
					RangeInclusive::new(f64::NEG_INFINITY, f64::INFINITY)
				},
			};
			if range.start() > range.end() {
				range = *range.end()..=*range.start();
			}
			let start = *range.start();
			let end = *range.end();

			ui.vertical(|ui| {
				ui.horizontal(|ui| {
					let mut v = value.to_f64();
					if v > end {
						v = start + (v - end);
						// v -= end - start;
						result.animating = true;
					} else if v < start {
						v = end - (start - v);
						// v += end - start;
						result.animating = true;
					}
					v = v.clamp(start, end);

					if full_width_slider(ui, &mut v, range, *step, T::EPSILON) {
						entry.active = false;
						// result.animating = true;
					}

					let new_value = T::from_f64(v);
					if original_value != new_value {
						*value = new_value;
						result.animating = true;
					}
				});
				// second row: type/start/end/step
				ui.horizontal(|ui| {
					ui.menu_button(ty.symbol(), |ui| {
						let lfab = ConstantType::LoopForwardAndBackward { forward: true };
						if ui.button(lfab.name()).clicked() {
							*ty = lfab;
							result.animating = true;
						}
						if ui.button(ConstantType::LoopForward.name()).clicked() {
							*ty = ConstantType::LoopForward;
							result.animating = true;
						}
						if ui.button(ConstantType::PlayOnce.name()).clicked() {
							*ty = ConstantType::PlayOnce;
							result.animating = true;
						}
						if ui.button(ConstantType::PlayIndefinitely.name()).clicked() {
							*ty = ConstantType::PlayIndefinitely;
							result.animating = true;
						}
					})
					.response
					.on_hover_text("Choose the constant animation type.");

					match ty {
						ConstantType::LoopForwardAndBackward { .. }
						| ConstantType::LoopForward
						| ConstantType::PlayOnce => {
							ui.label("Start: ");
							if let Err(e) = expr_ui(range_start, ui, "start", Some(80.0), clear_cache, false) {
								range_error = Some(e);
							}
							ui.label("End: ");
							if let Err(e) = expr_ui(range_end, ui, "end", Some(80.0), clear_cache, false) {
								range_error = Some(e);
							}
						},
						ConstantType::PlayIndefinitely => {
							ui.label("Start: ");
							if let Err(e) = expr_ui(range_start, ui, "start", Some(80.0), clear_cache, false) {
								range_error = Some(e);
							}
						},
					}

					let steps_step = *step * 0.1;
					let prev_step = *step;
					let step_range = T::EPSILON * 100.0..=10.0;
					let response = DragValue::new(step)
						.prefix("Step:")
						.speed(steps_step)
						.range(step_range.clone())
						.ui(ui);
					if response.dragged() {
						let max_change = steps_step * 2.0;
						let shift = ui.input(|i| i.modifiers.shift_only());

						let speed = steps_step.min(0.4);
						let speed = if shift { speed * 0.1 } else { speed };

						let mdelta = response.drag_delta();
						let delta_points = mdelta.x - mdelta.y; // Increase to the right and up
						let delta_value = delta_points as f64 * speed;
						if delta_value != 0.0 {
							*step = prev_step + delta_value.clamp(-max_change, max_change);
							*step = step.clamp(*step_range.start(), *step_range.end());
						}
					}

					if !prev_active && entry.active {
						if value.to_f64() >= end {
							*value = T::from_f64(start);
							result.animating = true;
						}
					}

					if !entry.active {
						return;
					}
					ui.ctx().request_repaint();
					result.animating = true;

					match ty {
						ConstantType::LoopForwardAndBackward { forward } => {
							if *forward {
								*value = *value + step_f;
							} else {
								*value = *value - step_f;
							}
							if value.to_f64() > end {
								*forward = false;
								*value = T::from_f64(end);
							}
							if value.to_f64() < start {
								*forward = true;
								*value = T::from_f64(start);
							}
						},
						ConstantType::LoopForward => {
							*value = *value + step_f;
							if value.to_f64() >= end {
								*value = T::from_f64(start);
							}
						},
						ConstantType::PlayOnce | ConstantType::PlayIndefinitely => {
							*value = *value + step_f;
							if value.to_f64() >= end {
								entry.active = false;
							}
						},
					}
				});
				if let Some(e) = range_error {
					ui.label(RichText::new(e).color(Color32::RED));
				}

				// *value = T::from_f64(v);
			});
		},
		EntryType::Color(color) => {
			ui.horizontal(|ui| {
				match expr_ui(&mut color.expr, ui, "(255,0,0)", None, clear_cache, true) {
					Ok(changed) => {
						result.parsed |= changed;
						result.needs_recompilation |= changed;
					},
					Err(e) => {
						result.error = Some(format!("Parsing error: {e}"));
					},
				}
				if let Some(color) = processed_colors.find_color(entry.id) {
					let mut stack = Stack::<T>::new();
					match  color.get_color32(&mut stack, ctx, T::ZERO, T::ZERO, T::ZERO) {
            Ok(color) => {
						ui.label(RichText::new("    ").background_color(color));
            }
            Err(e) => {
              result.error = Some(e);
            }
          }
				}
			});
		},
	}
}
fn is_alphanumeric_or_underscore(c: char) -> bool { c.is_alphanumeric() || c == '_' }
fn replace_symbols(ui: &egui::Ui, response: egui::Response, text: &mut String) {
	const REPLACEMENTS: &[(&str, char)] = &[
		("infinity", '∞'),
		("Infinity", '∞'),
		("pi", 'π'),
		("tau", 'τ'),
		("Integral", '∫'),
		("integral", '∫'),
		("Sum", '∑'),
		("sum", '∑'),
		("Product", '∏'),
		("product", '∏'),
	];

	let Some(mut state) = TextEditState::load(ui.ctx(), response.id) else {
		return;
	};
	let Some(cursor_range) = state.cursor.char_range() else {
		return;
	};
	let cursor_pos = cursor_range.primary.index;
	let text_len = text.len();

	for &(pattern, replacement) in REPLACEMENTS {
		let pattern_len = pattern.len();

		// check all possible positions
		let start_min = cursor_pos.saturating_sub(pattern_len + 1);
		let start_max = cursor_pos;
		for start in start_min..=start_max {
			let end = start + pattern_len;

			if end > text_len {
				continue;
			}
			let Some(slice) = text.get(start..end) else {
				continue;
			};
			if slice != pattern {
				continue;
			}

			// check char before pattern
			let valid_before = if start == 0 {
				true
			} else if let Some(before_char) = text[..start].chars().last() {
				!is_alphanumeric_or_underscore(before_char)
			} else {
				true
			};

			// check char after pattern
			let valid_after = if end >= text_len {
				true
			} else if let Some(after_char) = text[end..].chars().next() {
				!is_alphanumeric_or_underscore(after_char)
			} else {
				true
			};

			if !(valid_before && valid_after) {
				continue;
			}
			text.replace_range(start..end, &replacement.to_string());

			let new_cursor_pos = if cursor_pos <= start {
				start
			} else if cursor_pos > end {
				start + 2
			} else {
				start + 1
			};

			// update cursor
			let ccursor = egui::text::CCursor::new(new_cursor_pos);
			state.cursor.set_char_range(Some(CCursorRange::one(ccursor)));
			state.store(ui.ctx(), response.id);
			ui.ctx().memory_mut(|mem| mem.request_focus(response.id));
			return;
		}
	}
}
fn expr_ui<T: EvalexprFloat>(
	expr: &mut Expr<T>, ui: &mut egui::Ui, hint_text: &str, desired_width: Option<f32>, force_update: bool,
	preprocess: bool,
) -> Result<bool, String> {
	ui.horizontal_top(|ui| {
		let mut changed = false;
		let original_spacing = ui.style().spacing.item_spacing;
		ui.style_mut().spacing.item_spacing = vec2(0.0, 0.0);
		let mut text_edit = if let Some(desired_width) = desired_width {
			TextEdit::singleline(&mut expr.text).desired_width(desired_width).clip_text(false)
		} else {
			TextEdit::multiline(&mut expr.text).desired_rows(1).desired_width(ui.available_width())
		};
		// .min_size(vec2(desired_width.unwrap_or_else(|| ui.available_width()),0.0));
		// .desired_width(desired_width.unwrap_or_else(|| ui.available_width()));

		// let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap_width: f32| {
		// 	let mut layout_job: egui::text::LayoutJob = egui_extras::syntax_highlighting::highlight(
		// 		ui.ctx(),
		// 		ui.style(),
		// 		&CodeTheme::dark(12.0),
		// 		buf.as_str(),
		// 		"Rust",
		// 	);
		// 	layout_job.wrap.max_width = wrap_width;
		// 	ui.fonts_mut(|f| f.layout_job(layout_job))
		// };
		// text_edit = text_edit.layouter(&mut layouter);

		text_edit = text_edit.hint_text(hint_text);
		let response = ui.add(text_edit);
		if response.changed() || force_update {
			if expr.text.is_empty() {
				expr.node = None;
			} else {
				replace_symbols(ui, response, &mut expr.text);

				// let temp;
				// let mut txt = &expr.text;

				let mut ast = evalexpr::build_ast::<T>(&expr.text).map_err(|e| e.to_string())?;

				expr.node = None;
				expr.equation_type = EquationType::None;

				if preprocess {
					match preprocess_ast(ast) {
						Ok((new_ast, equation_type)) => {
							expr.equation_type = equation_type;
							ast = new_ast;
						},
						Err(e) => {
							return Err(e);
						},
					}
				}

				expr.node = match evalexpr::build_flat_node_from_ast::<T>(ast) {
					Ok(func) => Some(func),
					Err(e) => {
						return Err(e.to_string());
					},
				};
			}

			changed = true;
		}
		ui.style_mut().spacing.item_spacing = original_spacing;
		Ok(changed)
	})
	.inner
}

fn label_config_ui<T: EvalexprFloat>(label_config: &mut LabelConfig<T>, ui: &mut egui::Ui) -> bool {
	let mut changed = false;
	ui.horizontal(|ui| {
		ui.label("Text:");
		changed |= ui.text_edit_singleline(&mut label_config.text).changed();
	});
	changed |= ui.checkbox(&mut label_config.italic, "Italic").clicked();
	ui.horizontal(|ui| {
		SubMenuButton::new(format!("Size {:?}", label_config.size)).ui(ui, |ui| {
			use LabelSize as LS;
			changed |= ui
				.selectable_value(
					&mut label_config.size,
					LS::Small,
					RichText::new("Small").size(LS::Small.size()),
				)
				.changed();
			changed |= ui
				.selectable_value(
					&mut label_config.size,
					LS::Medium,
					RichText::new("Medium").size(LS::Medium.size()),
				)
				.changed();
			changed |= ui
				.selectable_value(
					&mut label_config.size,
					LS::Large,
					RichText::new("Large").size(LS::Large.size()),
				)
				.changed();
		});
	});
	ui.horizontal(|ui| {
		use LabelPosition as LP;
		SubMenuButton::new(format!("Position {}", label_config.pos.symbol())).ui(ui, |ui| {
			changed |= ui
				.selectable_value(&mut label_config.pos, LP::Top, format!("Top {}", LP::Top.symbol()))
				.changed();
			changed |= ui
				.selectable_value(&mut label_config.pos, LP::Bottom, format!("Bottom {}", LP::Bottom.symbol()))
				.changed();
			changed |= ui
				.selectable_value(&mut label_config.pos, LP::Left, format!("Left {}", LP::Left.symbol()))
				.changed();
			changed |= ui
				.selectable_value(&mut label_config.pos, LP::Right, format!("Right {}", LP::Right.symbol()))
				.changed();
			changed |= ui
				.selectable_value(&mut label_config.pos, LP::TLeft, format!("Top Left {}", LP::TLeft.symbol()))
				.changed();
			changed |= ui
				.selectable_value(
					&mut label_config.pos,
					LP::TRight,
					format!("Top Right {}", LP::TRight.symbol()),
				)
				.changed();
			changed |= ui
				.selectable_value(
					&mut label_config.pos,
					LP::BLeft,
					format!("Bottom Left {}", LP::BLeft.symbol()),
				)
				.changed();
			changed |= ui
				.selectable_value(
					&mut label_config.pos,
					LP::BRight,
					format!("Bottom Right {}", LP::BRight.symbol()),
				)
				.changed();
			changed |= ui
				.selectable_value(&mut label_config.pos, LP::Center, format!("Center {}", LP::Center.symbol()))
				.changed();
		});
	});
	match expr_ui(&mut label_config.angle, ui, "Angle", Some(80.0), false, false) {
		Ok(ch) => {
			changed |= ch;
		},
		Err(_) => {},
	}

	changed
}

fn line_style_config_ui(config: &mut LineStyleConfig, ui: &mut egui::Ui) -> bool {
	let mut changed = false;
	changed |= ui.checkbox(&mut config.selectable, "Lines Selectable").clicked();
	ui.separator();
	changed |= Slider::new(&mut config.line_width, 0.1..=10.0).text("Line Width").ui(ui).changed();
	ui.separator();
	ui.horizontal(|ui| {
		changed |= ui.selectable_value(&mut config.line_style, LineStyleType::Solid, "Solid").changed();
		changed |= ui.selectable_value(&mut config.line_style, LineStyleType::Dotted, "Dotted").changed();
		changed |= ui.selectable_value(&mut config.line_style, LineStyleType::Dashed, "Dashed").changed();
	});
	match &mut config.line_style {
		LineStyleType::Solid => {},
		LineStyleType::Dotted => {
			changed |= ui.add(Slider::new(&mut config.line_style_size, 0.1..=20.0).text("Spacing")).changed();
		},
		LineStyleType::Dashed => {
			changed |= ui.add(Slider::new(&mut config.line_style_size, 0.1..=20.0).text("Length")).changed();
		},
	}
	changed
}
fn fill_rule_btn_ui(ui: &mut egui::Ui, fill_rule: &mut FillRule) -> bool {
	let txt = if *fill_rule == FillRule::EvenOdd { "Even-Odd" } else { "Non-Zero" };
	let mut changed = false;
	ui.horizontal(|ui| {
		ui.label("Fill Rule:");
		ui.menu_button(txt, |ui| {
			ui.horizontal(|ui| {
				changed |= ui.selectable_value(fill_rule, FillRule::EvenOdd, "Even-Odd").changed();
				changed |= ui.selectable_value(fill_rule, FillRule::NonZero, "Non-Zero").changed();
			});
		});
	});
	changed
}
