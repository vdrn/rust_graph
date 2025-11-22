use core::ops::RangeInclusive;

use eframe::egui::containers::menu::{MenuButton, MenuConfig, SubMenuButton};
use eframe::egui::text::CCursorRange;
use eframe::egui::text_edit::TextEditState;
use eframe::egui::{
	self, Align, Button, Color32, DragValue, Label, PopupCloseBehavior, RichText, Slider, TextEdit, Widget, vec2
};

use evalexpr::{EvalexprFloat, HashMapContext};

use crate::entry::entry_processing::preprecess_fn;
use crate::entry::{
	COLORS, ConstantType, DragPoint, Entry, EntryType, Expr, LabelConfig, LabelPosition, LabelSize, LineStyleConfig, LineStyleType, MAX_IMPLICIT_RESOLUTION, MIN_IMPLICIT_RESOLUTION, PointDragType, PointEntry, RESERVED_NAMES, f64_to_float
};

pub struct EditEntryResult {
	pub needs_recompilation: bool,
	pub animating:           bool,
	pub remove:              bool,
	pub error:               Option<String>,
	pub parsed:              bool,
}
pub fn entry_ui<T: EvalexprFloat>(
	ui: &mut egui::Ui, ctx: &HashMapContext<T>, entry: &mut Entry<T>, clear_cache: bool,
) -> EditEntryResult {
	let mut result = EditEntryResult {
		needs_recompilation: false,
		animating:           false,
		remove:              false,
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
				Button::new(RichText::new(entry.symbol()).strong().monospace().color(text_col))
					.fill(fill_col)
					.corner_radius(10),
			)
			.clicked()
		{
			entry.active = !entry.active;
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
			entry_style(ui, entry);

			if ui.button("X").clicked() {
				result.remove = true;
				result.needs_recompilation = true;
			}
		});
		// entry edit
		ui.horizontal(|ui| {
			entry_type_ui(ui, ctx, entry, clear_cache, prev_visible, &mut result);
		});
	});

	result
}

fn entry_style<T: EvalexprFloat>(ui: &mut egui::Ui, entry: &mut Entry<T>) {
	let mut style_button = MenuButton::new(RichText::new("🎨").color(Color32::BLACK))
		.config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside));
	style_button.button = style_button.button.fill(entry.color());
	style_button.ui(ui, |ui| {
		let mut color_button = SubMenuButton::new(RichText::new("Color").color(Color32::BLACK));
		color_button.button = color_button.button.fill(entry.color());

		color_button.ui(ui, |ui| {
			for i in 0..COLORS.len() {
				if ui.button(RichText::new("     ").background_color(COLORS[i])).clicked() {
					entry.color = i;
				}
			}
		});
		match &mut entry.ty {
			EntryType::Function { style, .. } => {
				ui.separator();

				line_style_config_ui(style, ui);
				ui.separator();
				egui::Sides::new().show(
					ui,
					|_ui| {},
					|ui| {
						if ui.button("Close").clicked() {
							ui.close();
						}
						if ui.button("Reset").clicked() {
							*style = Default::default();
							ui.close();
						}
					},
				);
			},
			EntryType::Points { style, .. } => {
				ui.separator();
				let mut show_label = style.label_config.is_some();
				if ui.checkbox(&mut show_label, "Show label").changed() {
					if show_label {
						style.label_config = Some(LabelConfig::default());
					} else {
						style.label_config = None;
					}
				}
				if let Some(label_config) = &mut style.label_config {
					label_config_ui(label_config, ui);
				}
				ui.separator();
				ui.checkbox(&mut style.show_lines, "Show Lines");
				if style.show_lines {
					ui.label("Line Style:");
					ui.checkbox(&mut style.show_arrows, "Show Arrows");
					line_style_config_ui(&mut style.line_style, ui);
				} else {
					style.show_arrows = false;
				}
				ui.separator();
				ui.checkbox(&mut style.show_points, "Show Not Draggable Points");

				egui::Sides::new().show(
					ui,
					|_ui| {},
					|ui| {
						if ui.button("Close").clicked() {
							ui.close();
						}
						if ui.button("Reset").clicked() {
							*style = Default::default();
							ui.close();
						}
					},
				);
			},
			EntryType::Constant { .. } => {},
			EntryType::Label { .. } => {},
			EntryType::Folder { .. } => {},
		}
	});
}
fn entry_type_ui<T: EvalexprFloat>(
	ui: &mut egui::Ui, ctx: &HashMapContext<T>, entry: &mut Entry<T>, clear_cache: bool, prev_active: bool,
	result: &mut EditEntryResult,
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
					if let Some(computed_const) = func.computed_const() {
						ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
							Label::new(computed_const.human_display(func.display_rational))
								.selectable(true)
								.ui(ui);

							ui.label("=");
							if Button::new(if func.display_rational { "Q" } else { "F" }).ui(ui).clicked() {
								func.display_rational = !func.display_rational;
							}
						});
					}
					if func.args.len() == 1 {
						ui.horizontal(|ui| {
							ui.checkbox(parametric, "Parametric");
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
					}
					if func.args.len() == 2 && func.args[0].to_str() == "x" && func.args[1].to_str() == "y" {
						ui.horizontal(|ui| {
							Slider::new(
								implicit_resolution,
								MIN_IMPLICIT_RESOLUTION..=MAX_IMPLICIT_RESOLUTION,
							)
							.text("Implicit Resolution")
							.ui(ui);
						});
					}
				});
			});
		},
		EntryType::Label { x, y, size, underline } => {
			ui.horizontal(|ui| {
				match expr_ui(x, ui, "point_x", Some(80.0), clear_cache, false) {
					Ok(changed) => {
						result.parsed |= changed;
						// result.needs_recompilation |= changed;
					},
					Err(e) => {
						result.error = Some(format!("Parsing error: {e}"));
					},
				}
				match expr_ui(y, ui, "point_y", Some(80.0), clear_cache, false) {
					Ok(changed) => {
						result.parsed |= changed;
						// result.needs_recompilation |= changed;
					},
					Err(e) => {
						result.error = Some(format!("Parsing error: {e}"));
					},
				}
				match expr_ui(size, ui, "size", Some(80.0), clear_cache, false) {
					Ok(changed) => {
						result.parsed |= changed;
						// result.needs_recompilation |= changed;
					},
					Err(e) => {
						result.error = Some(format!("Parsing error: {e}"));
					},
				}
				ui.checkbox(underline, "Underline");
			});
		},
		EntryType::Points { points, .. } => {
			let mut remove_point = None;
			ui.vertical(|ui| {
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
						if !point.both_drag_dirs_available && point.drag_type == PointDragType::Both {
							point.both_drag_dirs_available = false;
							drag_type_changed = true;
						}

						let drag_menu_text = match &point.drag_point {
							Some(d) => match d {
								DragPoint::BothCoordLiterals => {
									format!("{}(x,y)", point.drag_type.symbol())
								},
								DragPoint::XLiteral => format!("{}(x,_)", point.drag_type.symbol()),
								DragPoint::YLiteral => format!("{}(_,y)", point.drag_type.symbol()),
								DragPoint::XConstant(x) => {
									format!("{}({},_)", point.drag_type.symbol(), x)
								},
								DragPoint::YConstant(y) => {
									format!("{}(_, {})", point.drag_type.symbol(), y)
								},
								DragPoint::XLiteralYConstant(y) => {
									format!("{}(x, {})", point.drag_type.symbol(), y)
								},
								DragPoint::YLiteralXConstant(x) => {
									format!("{}({}, y)", point.drag_type.symbol(), x)
								},
								DragPoint::BothCoordConstants(x, y) => {
									format!("{}({}, {})", point.drag_type.symbol(), x, y,)
								},
								DragPoint::SameConstantBothCoords(x) => {
									format!("{}({})", point.drag_type.symbol(), x)
								},
							},
							None => point.drag_type.symbol().to_string(),
						};
						ui.menu_button(drag_menu_text, |ui| {
							drag_type_changed |= ui
								.selectable_value(
									&mut point.drag_type,
									PointDragType::NoDrag,
									PointDragType::NoDrag.name(),
								)
								.changed();
							if point.both_drag_dirs_available {
								drag_type_changed |= ui
									.selectable_value(
										&mut point.drag_type,
										PointDragType::Both,
										PointDragType::Both.name(),
									)
									.changed();
							}
							drag_type_changed |= ui
								.selectable_value(
									&mut point.drag_type,
									PointDragType::X,
									PointDragType::X.name(),
								)
								.changed();
							drag_type_changed |= ui
								.selectable_value(
									&mut point.drag_type,
									PointDragType::Y,
									PointDragType::Y.name(),
								)
								.changed();
						});
						if drag_type_changed {
							result.parsed = true;
							result.needs_recompilation = true;
						}

						if ui.button("❌").clicked() {
							remove_point = Some(pi);
						}
					});
				}
				if let Some(pi) = remove_point {
					points.remove(pi);
				}
				ui.horizontal(|ui| {
					if ui.button("Add Point").clicked() {
						points.push(PointEntry::default());
					}
				});
			});
		},

		EntryType::Constant { value, step, ty, range_start, range_end, .. } => {
			let original_value = *value;
			let step_f = f64_to_float::<T>(*step);

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

					if slider_full_width(ui, &mut v, range, *step) {
						entry.active = false;
					}

					if original_value.to_f64() != v {
						*value = f64_to_float::<T>(v);
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
					});

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
							if let Err(e) = expr_ui(range_start, ui, "start", Some(80.0), clear_cache, false){
                range_error = Some(e);
              }
						},
					}

					DragValue::new(step).prefix("Step:").speed(0.00001).ui(ui);

					if !prev_active && entry.active {
						if value.to_f64() >= end {
							*value = f64_to_float::<T>(start);
							result.animating = true;
						}
					}

					if !entry.active {
						return;
					}
					ui.ctx().request_repaint();
					result.animating = true;

					match ty {
						ConstantType::LoopForwardAndBackward { forward, .. } => {
							if *forward {
								*value = *value + step_f;
							} else {
								*value = *value - step_f;
							}
							if value.to_f64() > end {
								*forward = false;
								*value = f64_to_float::<T>(end);
							}
							if value.to_f64() < start {
								*forward = true;
								*value = f64_to_float::<T>(start);
							}
						},
						ConstantType::LoopForward { .. } => {
							*value = *value + step_f;
							if value.to_f64() >= end {
								*value = f64_to_float::<T>(start);
							}
						},
						ConstantType::PlayOnce { .. } | ConstantType::PlayIndefinitely { .. } => {
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

				// *value = f64_to_float::<T>(v);
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
		let mut text_edit = TextEdit::multiline(&mut expr.text)
			.desired_rows(1)
			.desired_width(desired_width.unwrap_or_else(|| ui.available_width()));

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

				let temp;
				let mut txt = &expr.text;
				if preprocess {
					match preprecess_fn(&expr.text) {
						Ok(Some(new_text)) => {
							temp = new_text;
							txt = &temp;
						},
						Err(e) => {
							return Err(e);
						},
						_ => {},
					}
				}

				expr.node = match evalexpr::build_operator_tree::<T>(txt) {
					Ok(func) => Some(func),
					Err(e) => {
						return Err(e.to_string());
					},
				};
				expr.inlined_node = None;
			}

			changed = true;
		}
		ui.style_mut().spacing.item_spacing = original_spacing;
		Ok(changed)
	})
	.inner
}

fn label_config_ui(label_config: &mut LabelConfig, ui: &mut egui::Ui) {
	ui.checkbox(&mut label_config.italic, "Italic");
	ui.horizontal(|ui| {
		SubMenuButton::new(format!("Size {:?}", label_config.size)).ui(ui, |ui| {
			use LabelSize as LS;
			ui.selectable_value(
				&mut label_config.size,
				LS::Small,
				RichText::new("Small").size(LS::Small.size()),
			);
			ui.selectable_value(
				&mut label_config.size,
				LS::Medium,
				RichText::new("Medium").size(LS::Medium.size()),
			);
			ui.selectable_value(
				&mut label_config.size,
				LS::Large,
				RichText::new("Large").size(LS::Large.size()),
			);
		});
	});
	ui.horizontal(|ui| {
		use LabelPosition as LP;
		SubMenuButton::new(format!("Position {}", label_config.pos.symbol())).ui(ui, |ui| {
			ui.selectable_value(&mut label_config.pos, LP::Top, format!("Top {}", LP::Top.symbol()));
			ui.selectable_value(&mut label_config.pos, LP::Bottom, format!("Bottom {}", LP::Bottom.symbol()));
			ui.selectable_value(&mut label_config.pos, LP::Left, format!("Left {}", LP::Left.symbol()));
			ui.selectable_value(&mut label_config.pos, LP::Right, format!("Right {}", LP::Right.symbol()));
			ui.selectable_value(&mut label_config.pos, LP::TLeft, format!("Top Left {}", LP::TLeft.symbol()));
			ui.selectable_value(
				&mut label_config.pos,
				LP::TRight,
				format!("Top Right {}", LP::TRight.symbol()),
			);
			ui.selectable_value(
				&mut label_config.pos,
				LP::BLeft,
				format!("Bottom Left {}", LP::BLeft.symbol()),
			);
			ui.selectable_value(
				&mut label_config.pos,
				LP::BRight,
				format!("Bottom Right {}", LP::BRight.symbol()),
			);
		});
	});
}

fn line_style_config_ui(config: &mut LineStyleConfig, ui: &mut egui::Ui) {
	Slider::new(&mut config.line_width, 0.1..=10.0).text("Line Width").ui(ui);
	ui.separator();
	ui.horizontal(|ui| {
		ui.selectable_value(&mut config.line_style, LineStyleType::Solid, "Solid");
		ui.selectable_value(&mut config.line_style, LineStyleType::Dotted, "Dotted");
		ui.selectable_value(&mut config.line_style, LineStyleType::Dashed, "Dashed");
	});
	match &mut config.line_style {
		LineStyleType::Solid => {},
		LineStyleType::Dotted => {
			ui.add(Slider::new(&mut config.line_style_size, 0.1..=20.0).text("Spacing"));
		},
		LineStyleType::Dashed => {
			ui.add(Slider::new(&mut config.line_style_size, 0.1..=20.0).text("Length"));
		},
	}
}

fn slider_full_width<Num: egui::emath::Numeric>(
	ui: &mut egui::Ui, value: &mut Num, range: core::ops::RangeInclusive<Num>, step: f64,
) -> bool {
	let mut changed = false;
	// ui.with_layout(egui::Layout::centered_and_justified(egui::Direction::RightToLeft), |ui| {
	if DragValue::new(value).speed(step * 0.5).ui(ui).changed() {
		changed = true;
	}
	let default_slider_width = ui.style().spacing.slider_width;
	ui.style_mut().spacing.slider_width = ui.available_width();
	if ui
		.add(Slider::new(value, range).step_by(step).clamping(egui::SliderClamping::Never).show_value(false))
		.dragged()
	{
		changed = true;
	}
	ui.style_mut().spacing.slider_width = default_slider_width;
	// });
	changed
}
