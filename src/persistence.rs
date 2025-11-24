use alloc::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use base64::Engine;
use eframe::egui::{self, Grid, Id, Modal};
use egui_plot::PlotBounds;
use evalexpr::{EvalexprFloat, istr, istr_empty};
use serde::{Deserialize, Serialize};

use crate::entry::{
	EquationType, Expr, FunctionType, LineStyleConfig, MAX_IMPLICIT_RESOLUTION, MIN_IMPLICIT_RESOLUTION, PointDragType, PointStyle, preprocess_ast
};
use crate::{ConstantType, Entry, EntryType, PointEntry, State, UiState};

pub fn default_true() -> bool { true }

#[derive(Serialize, Deserialize)]
pub struct StateSerialized {
	pub entries:        Vec<EntrySerialized>,
	#[serde(default)]
	pub default_bounds: Option<[f64; 4]>,
}

#[derive(Serialize, Deserialize)]
pub struct EntrySerialized {
	#[serde(default)]
	name:    String,
	#[serde(default = "default_true")]
	visible: bool,
	color:   usize,
	ty:      EntryTypeSerialized,
}
#[derive(Serialize, PartialEq, Deserialize, Default)]
pub struct ExprSer {
	#[serde(default)]
	text:             String,
	#[serde(default)]
	display_rational: bool,
}
impl ExprSer {
	pub fn from_expr<T: EvalexprFloat>(expr: &Expr<T>) -> Self {
		Self { text: expr.text.clone(), display_rational: expr.display_rational }
	}
	pub fn into_expr<T: EvalexprFloat>(self, preprocess: bool) -> Expr<T> {
		let mut equation_type = EquationType::None;
		let mut ast = evalexpr::build_ast::<T>(&self.text).ok();
		if preprocess
			&& let Some((new_ast, new_equation_type)) = ast.take().and_then(|ast| preprocess_ast(ast).ok())
		{
			ast = Some(new_ast);
			equation_type = new_equation_type;
		}
		Expr {
			node: ast.and_then(|ast| evalexpr::build_flat_node_from_ast::<T>(ast).ok()),
			equation_type,

			inlined_node: None,
			args: Vec::new(),
			expr_function: None,
			text: self.text,
			display_rational: self.display_rational,
		}
	}
}
#[derive(Serialize, Deserialize)]
pub enum EntryTypeSerialized {
	Function {
		func:                ExprSer,
		#[serde(default)]
		ranged:              bool,
		#[serde(default)]
		range_start:         ExprSer,
		#[serde(default)]
		range_end:           ExprSer,
		#[serde(default)]
		style:               LineStyleConfig,
		#[serde(default)]
		implicit_resolution: usize,
	},
	Constant {
		value:       f64,
		step:        f64,
		ty:          ConstantType,
		#[serde(default)]
		range_start: ExprSer,
		#[serde(default)]
		range_end:   ExprSer,
	},
	Points {
		points: Vec<EntryPointSerialized>,
		#[serde(default)]
		style:  PointStyle,
	},
	Label {
		#[serde(default)]
		x:         ExprSer,
		#[serde(default)]
		y:         ExprSer,
		#[serde(default)]
		size:      ExprSer,
		#[serde(default)]
		underline: bool,
	},
	Folder {
		#[serde(default)]
		entries: Vec<EntrySerialized>,
	},
}
#[derive(Serialize, Deserialize)]
pub struct EntryPointSerialized {
	#[serde(default)]
	x:         ExprSer,
	#[serde(default)]
	y:         ExprSer,
	#[serde(default)]
	drag_type: PointDragType,
}

pub fn entries_to_ser<T: EvalexprFloat>(
	entries: &[Entry<T>], plot_bounds: Option<&PlotBounds>,
) -> StateSerialized {
	let mut state_serialized = StateSerialized {
		entries:        Vec::with_capacity(entries.len()),
		default_bounds: plot_bounds.map(|b| [b.min()[0], b.min()[1], b.max()[0], b.max()[1]]),
	};
	for entry in entries {
		let entry_serialized = EntrySerialized {
			name:    entry.name.clone(),
			visible: entry.active,
			color:   entry.color,
			ty:      match &entry.ty {
				EntryType::Function {
					func,
					range_start,
					range_end,
					style,
					parametric,
					implicit_resolution,
					..
				} => EntryTypeSerialized::Function {
					func:                ExprSer::from_expr(func),
					ranged:              *parametric,
					range_start:         ExprSer::from_expr(range_start),
					range_end:           ExprSer::from_expr(range_end),
					style:               style.clone(),
					implicit_resolution: *implicit_resolution,
				},
				EntryType::Constant { value, step, ty, istr_name: _, range_start, range_end } => {
					EntryTypeSerialized::Constant {
						value:       value.to_f64(),
						step:        *step,
						ty:          ty.clone(),
						range_start: ExprSer::from_expr(range_start),
						range_end:   ExprSer::from_expr(range_end),
					}
				},
				EntryType::Points { points, style } => {
					let mut points_serialized = Vec::new();
					for point in points {
						let point_serialized = EntryPointSerialized {
							x:         ExprSer::from_expr(&point.x),
							y:         ExprSer::from_expr(&point.y),
							drag_type: point.drag_type,
						};
						points_serialized.push(point_serialized);
					}
					EntryTypeSerialized::Points { points: points_serialized, style: style.clone() }
				},
				EntryType::Label { x, y, size, underline, .. } => EntryTypeSerialized::Label {
					x:         ExprSer::from_expr(x),
					y:         ExprSer::from_expr(y),
					size:      ExprSer::from_expr(size),
					underline: *underline,
				},
				EntryType::Folder { entries } => {
					let ser_entries = entries_to_ser(entries, None);
					EntryTypeSerialized::Folder { entries: ser_entries.entries }
				},
			},
		};
		state_serialized.entries.push(entry_serialized);
	}
	state_serialized
}
pub fn serialize_to_json<T: EvalexprFloat>(
	writer: impl Write, state: &[Entry<T>], plot_bounds: Option<&PlotBounds>,
) -> std::io::Result<()> {
	let ser = entries_to_ser(state, plot_bounds);
	serde_json::to_writer(writer, &ser)?;
	Ok(())
}
pub fn serialize_to_url<T: EvalexprFloat>(
	entries: &[Entry<T>], plot_bounds: Option<&PlotBounds>,
) -> Result<String, String> {
	let ser = entries_to_ser(entries, plot_bounds);
	let bincoded =
		bincode::serde::encode_to_vec(&ser, bincode::config::standard()).map_err(|e| e.to_string())?;
	let base64_encoded = base64::engine::general_purpose::STANDARD.encode(bincoded);
	Ok(base64_encoded)
}

#[cfg(target_arch = "wasm32")]
pub fn deserialize_from_url<T: EvalexprFloat>(
	id_counter: &mut u64,
) -> Result<(Vec<Entry<T>>, Option<PlotBounds>), String> {
	let href = web_sys::window()
		.expect("Couldn't get window")
		.document()
		.expect("Couldn't get document")
		.location()
		.expect("Couldn't get location")
		.href()
		.expect("Couldn't get href");
	// let href = "";

	if !href.contains('#') {
		return Ok((Vec::new(), None));
	}
	let Some(without_prefix) = href.split('#').last() else {
		return Ok((Vec::new(), None));
	};

	let base64_decoded =
		base64::engine::general_purpose::STANDARD.decode(without_prefix).map_err(|e| e.to_string())?;
	// let base64_decoded = base64::decode(without_prefix).map_err(|e| e.to_string())?;
	let entries_ser = bincode::serde::decode_from_slice(&base64_decoded, bincode::config::standard())
		.map_err(|e| e.to_string())?
		.0;
	Ok(entries_from_ser(entries_ser, id_counter))

	// let decoded = urlencoding::decode(without_prefix).map_err(|e| e.to_string())?;
	// deserialize_from_json(decoded.as_bytes())
}

pub fn entries_from_ser<T: EvalexprFloat>(
	ser: StateSerialized, id: &mut u64,
) -> (Vec<Entry<T>>, Option<PlotBounds>) {
	let mut result = Vec::new();
	let bounds = ser.default_bounds.map(|b| PlotBounds::from_min_max([b[0], b[1]], [b[2], b[3]]));
	for entry in ser.entries {
		*id += 1;
		let entry_deserialized = Entry {
			id:     *id,
			active: entry.visible,
			color:  entry.color,
			ty:     match entry.ty {
				EntryTypeSerialized::Function {
					func,
					ranged,
					range_start,
					range_end,
					style,
					implicit_resolution,
				} => EntryType::Function {
					parametric: ranged,
					identifier: istr_empty(),
					func: func.into_expr(true),
					// Actual type will be set in compilation step later
					ty: FunctionType::Expression,
					can_be_drawn: true,
					// ty: if ranged { FunctionType::Ranged } else { FunctionType::X },
					range_start: range_start.into_expr(false),
					range_end: range_end.into_expr(false),
					implicit_resolution: implicit_resolution
						.clamp(MIN_IMPLICIT_RESOLUTION, MAX_IMPLICIT_RESOLUTION),

					style,
				},
				EntryTypeSerialized::Constant { value, step, ty, range_start, range_end } => {
					EntryType::Constant {
						value: T::from_f64(value),
						step,
						ty,
						istr_name: istr(&entry.name),
						range_start: range_start.into_expr(false),
						range_end: range_end.into_expr(false),
					}
				},
				EntryTypeSerialized::Points { points, style } => {
					let mut points_deserialized = Vec::new();
					for point in points {
						let point_deserialized = PointEntry {
							x:                        point.x.into_expr(false),
							y:                        point.y.into_expr(false),
							drag_point:               None,
							drag_type:                point.drag_type,
							both_drag_dirs_available: true,
							val:                      None,
						};
						points_deserialized.push(point_deserialized);
					}
					EntryType::Points { points: points_deserialized, style }
				},
				EntryTypeSerialized::Label { x: text_x, y: text_y, size, underline } => EntryType::Label {
					x: text_x.into_expr(false),
					y: text_y.into_expr(false),
					size: size.into_expr(false),
					underline,
				},
				EntryTypeSerialized::Folder { entries } => {
					let (entries, _) = entries_from_ser(StateSerialized { entries, default_bounds: None }, id);
					EntryType::Folder { entries }
				},
			},
			name:   entry.name,
		};

		result.push(entry_deserialized);
	}
	*id += 1;
	(result, bounds)
}
pub fn deserialize_from_json<T: EvalexprFloat>(
	reader: &[u8], id_counter: &mut u64,
) -> Result<(Vec<Entry<T>>, Option<PlotBounds>), String> {
	let entries: StateSerialized = serde_json::from_slice(reader).map_err(|e| e.to_string())?;
	Ok(entries_from_ser(entries, id_counter))
}
// pub fn deserialize_from_url<T: EvalexprFloat>(url: &str) -> Result<Vec<Entry<T>>, String> {
// }
#[cfg(target_arch = "wasm32")]
pub fn save_file<T: EvalexprFloat>(ui_state: &mut UiState, state: &State<T>, frame: &mut eframe::Frame) {
	let file = format!("{}.json", state.name);
	let mut output = Vec::new();
	if let Err(e) = serialize_to_json(&mut output, &state.entries, Some(&ui_state.plot_bounds)) {
		ui_state.serialization_error = Some(e.to_string());
	} else {
		ui_state.serialization_error = None;
		ui_state.serialized_states.insert(file, String::from_utf8(output).unwrap());
		if let Some(storage) = frame.storage_mut() {
			storage.flush();
		}
	}
}
#[cfg(not(target_arch = "wasm32"))]
pub fn save_file<T: EvalexprFloat>(ui_state: &mut UiState, state: &State<T>, _frame: &mut eframe::Frame) {
	use std::path::PathBuf;

	let save_path = PathBuf::from(&ui_state.cur_dir).join(format!("{}.json", state.name));
	if let Some(parent) = save_path.parent() {
		// Recursively create all parent directories if they don't exist

		if std::fs::create_dir_all(parent).is_err() {
			ui_state.serialization_error = Some(format!("Could not create directory: {}", parent.display()));
			return;
		}
	}
	let Ok(mut file) = std::fs::File::create(&save_path) else {
		ui_state.serialization_error = Some(format!("Could not create file: {}", save_path.display()));
		return;
	};
	if let Err(e) = serialize_to_json(&mut file, &state.entries, Some(&ui_state.plot_bounds)) {
		ui_state.serialization_error = Some(e.to_string());
	} else {
		ui_state.serialization_error = None;
		load_file_entries(&ui_state.cur_dir, &mut ui_state.serialized_states);
	}
}
pub fn load_file_entries(cur_dir: &str, ser_states: &mut BTreeMap<String, String>) {
	ser_states.clear();
	let Ok(entries) = std::fs::read_dir(PathBuf::from(cur_dir)) else {
		// ui.label("No entries found");
		return;
	};
	for entry in entries {
		let Ok(entry) = entry else {
			continue;
		};
		let file_name = entry.file_name();
		let Some(file_name) = file_name.to_str() else {
			continue;
		};
		if !file_name.ends_with(".json") {
			continue;
		}

		ser_states.insert(file_name.to_string(), String::new());
	}
}
#[cfg(target_arch = "wasm32")]
pub fn load_file<T: EvalexprFloat>(
	cur_dir: &str, ser_states: &BTreeMap<String, String>, file_name: &str, state: &mut State<T>,
	id_counter: &mut u64,
) -> Result<(), String> {
	if let Some(file) = ser_states.get(file_name) {
		let (entries, bounds) = deserialize_from_json::<T>(file.as_bytes(), id_counter)?;
		state.entries = entries;
		state.default_bounds = bounds;

		state.name = file_name.strip_suffix(".json").unwrap_or(file_name).to_string();
		state.clear_cache = true;
	}
	Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn load_file<T: EvalexprFloat>(
	cur_dir: &str, _ser_states: &BTreeMap<String, String>, file_name: &str, state: &mut State<T>,
	id_counter: &mut u64,
) -> Result<(), String> {
	let Ok(file) = std::fs::read(PathBuf::from(cur_dir).join(file_name)) else {
		return Err(format!("Could not open file: {}", file_name));
	};
	let (entries, default_bounds) = deserialize_from_json::<T>(&file, id_counter)
		.map_err(|e| format!("Could not deserialize file: {}", e))?;
	state.entries = entries;
	state.default_bounds = default_bounds;

	state.name = file_name.strip_suffix(".json").unwrap_or(file_name).to_string();
	state.clear_cache = true;

	Ok(())
}

pub fn persistence_ui<T: EvalexprFloat>(
	state: &mut State<T>, ui_state: &mut UiState, ui: &mut egui::Ui, frame: &mut eframe::Frame,
) {
	ui.horizontal_top(|ui| {
		ui.label("Name:");
		ui.text_edit_singleline(&mut state.name);
		if !state.name.trim().is_empty() && ui.button("Save").clicked() {
			save_file(ui_state, state, frame);
		}
	});

	#[cfg(not(target_arch = "wasm32"))]
	{
		ui.separator();
		ui.horizontal_top(|ui| {
			ui.label("CWD:");
			let changed = ui.text_edit_singleline(&mut ui_state.cur_dir).changed();

			if ui.button("⟳").clicked() || changed {
				load_file_entries(&ui_state.cur_dir, &mut ui_state.serialized_states);
			}
		});
	}

	if !ui_state.serialized_states.is_empty() {
		ui.separator();
	}
	ui.horizontal(|ui| {
		Grid::new("files").num_columns(3).striped(true).show(ui, |ui| {
			for file_name in ui_state.serialized_states.keys() {
				ui.label(file_name);
				if ui.button("Load").clicked() {
					if let Err(e) = load_file(
						&ui_state.cur_dir, &ui_state.serialized_states, file_name, state,
						&mut ui_state.next_id,
					) {
						ui_state.serialization_error = Some(format!("Could not open file: {}", e));
						return;
					};
					ui_state.serialization_error = None;
					ui_state.reset_graph = true;
				}
				if ui.button("Delete").clicked() {
					ui_state.file_to_remove = Some(file_name.clone());
				}
				ui.end_row();
			}
			confirm_remove_dialog(ui, frame, ui_state);
		});
	});
}

fn confirm_remove_dialog(ui: &egui::Ui, _frame: &mut eframe::Frame, ui_state: &mut UiState) {
	if let Some(file) = &ui_state.file_to_remove {
		let modal = Modal::new(Id::new("Confirm remove file")).show(ui.ctx(), |ui| {
			ui.set_width(400.0);
			ui.heading(format!("Are you sure you want to delete '{file}'?"));

			ui.add_space(32.0);

			egui::Sides::new().show(
				ui,
				|_ui| {},
				|ui| {
					if ui.button("Yes").clicked() {
						#[cfg(target_arch = "wasm32")]
						{
							if let Some(storage) = _frame.storage_mut() {
								storage.flush();
							}
							ui_state.serialized_states.remove(file);
						}
						#[cfg(not(target_arch = "wasm32"))]
						{
							if let Ok(()) = std::fs::remove_file(PathBuf::from(&ui_state.cur_dir).join(file)) {
								ui_state.serialized_states.remove(file);
							}
						}
						ui.close();
					}

					if ui.button("No").clicked() {
						ui.close();
					}
				},
			);
		});

		if modal.should_close() {
			ui_state.file_to_remove = None;
		}
	}
}
