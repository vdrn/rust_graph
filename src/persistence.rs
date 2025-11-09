use std::io::Write;
use std::path::PathBuf;

use base64::Engine;
use eframe::egui::{self, Grid, Id, Modal};
use egui_plot::PlotBounds;
use evalexpr::{EvalexprFloat, EvalexprNumericTypes};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::entry::{
  PointDragType,
	Expr, FunctionStyle, FunctionType, MAX_IMPLICIT_RESOLUTION, MIN_IMPLICIT_RESOLUTION, TextboxType
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
	text:         String,
	#[serde(default)]
	textbox_type: TextboxType,
}
impl ExprSer {
	pub fn from_expr<T: EvalexprNumericTypes>(expr: &Expr<T>) -> Self {
		Self { text: expr.text.clone(), textbox_type: expr.textbox_type }
	}
	pub fn into_expr<T: EvalexprNumericTypes>(self) -> Expr<T> {
		Expr {
			node:         evalexpr::build_operator_tree::<T>(&self.text).ok(),
			text:         self.text,
			textbox_type: self.textbox_type,
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
		style:               FunctionStyle,
		#[serde(default)]
		implicit_resolution: usize,
	},
	Constant {
		value: f64,
		step:  f64,
		ty:    ConstantType,
	},
	Points(Vec<EntryPointSerialized>),
	Integral {
		#[serde(default)]
		func:  ExprSer,
		#[serde(default)]
		lower: ExprSer,
		#[serde(default)]
		upper: ExprSer,

		resolution: usize,
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
}
#[derive(Serialize, Deserialize)]
pub struct EntryPointSerialized {
	#[serde(default)]
	x: ExprSer,
	#[serde(default)]
	y: ExprSer,
	#[serde(default)]
  drag_type: PointDragType,
}

pub fn entries_to_ser<T: EvalexprNumericTypes>(entries: &[Entry<T>], plot_bounds: Option<&PlotBounds>) -> StateSerialized {
	let mut state_serialized = StateSerialized {
		entries:        Vec::with_capacity(entries.len()),
		default_bounds: plot_bounds.map(|b| [b.min()[0], b.min()[1], b.max()[0], b.max()[1]]),
	};
	for entry in entries {
		let entry_serialized = EntrySerialized {
			name:    entry.name.clone(),
			visible: entry.visible,
			color:   entry.color,
			ty:      match &entry.ty {
				EntryType::Function {
					func, range_start, range_end, style, implicit_resolution, ty, ..
				} => EntryTypeSerialized::Function {
					func:                ExprSer::from_expr(func),
					ranged:              ty == &FunctionType::Ranged,
					range_start:         ExprSer::from_expr(range_start),
					range_end:           ExprSer::from_expr(range_end),
					style:               style.clone(),
					implicit_resolution: *implicit_resolution,
				},
				EntryType::Constant { value, step, ty } => {
					EntryTypeSerialized::Constant { value: value.to_f64(), step: *step, ty: ty.clone() }
				},
				EntryType::Points(points) => {
					let mut points_serialized = Vec::new();
					for point in points {
						let point_serialized = EntryPointSerialized {
							x: ExprSer::from_expr(&point.x),
							y: ExprSer::from_expr(&point.y),
              drag_type: point.drag_type,
						};
						points_serialized.push(point_serialized);
					}
					EntryTypeSerialized::Points(points_serialized)
				},
				EntryType::Integral { func, lower, upper, resolution, .. } => EntryTypeSerialized::Integral {
					func:       ExprSer::from_expr(func),
					lower:      ExprSer::from_expr(lower),
					upper:      ExprSer::from_expr(upper),
					resolution: *resolution,
				},
				EntryType::Label { x, y, size, underline, .. } => EntryTypeSerialized::Label {
					x:         ExprSer::from_expr(x),
					y:         ExprSer::from_expr(y),
					size:      ExprSer::from_expr(size),
					underline: *underline,
				},
			},
		};
		state_serialized.entries.push(entry_serialized);
	}
	state_serialized
}
pub fn serialize_to_json<T: EvalexprNumericTypes>(
	writer: impl Write, state: &[Entry<T>], plot_bounds: Option<&PlotBounds>,
) -> std::io::Result<()> {
	let ser = entries_to_ser(state, plot_bounds);
	serde_json::to_writer(writer, &ser)?;
	Ok(())
}
pub fn serialize_to_url<T: EvalexprNumericTypes>(entries: &[Entry<T>], plot_bounds: Option<&PlotBounds>) -> Result<String, String> {
	let ser = entries_to_ser(entries, plot_bounds);
	let bincoded =
		bincode::serde::encode_to_vec(&ser, bincode::config::standard()).map_err(|e| e.to_string())?;
	let base64_encoded = base64::engine::general_purpose::STANDARD.encode(bincoded);
	Ok(base64_encoded)
}

#[cfg(target_arch = "wasm32")]
pub fn deserialize_from_url<T: EvalexprNumericTypes>() -> Result<(Vec<Entry<T>>, Option<PlotBounds>), String> {
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
		return Ok((Vec::new(),None));
	}
	let Some(without_prefix) = href.split('#').last() else {
		return Ok((Vec::new(),None));
	};

	let base64_decoded =
		base64::engine::general_purpose::STANDARD.decode(without_prefix).map_err(|e| e.to_string())?;
	// let base64_decoded = base64::decode(without_prefix).map_err(|e| e.to_string())?;
	let entries_ser = bincode::serde::decode_from_slice(&base64_decoded, bincode::config::standard())
		.map_err(|e| e.to_string())?
		.0;
	Ok(entries_from_ser(entries_ser))

	// let decoded = urlencoding::decode(without_prefix).map_err(|e| e.to_string())?;
	// deserialize_from_json(decoded.as_bytes())
}

pub fn entries_from_ser<T: EvalexprNumericTypes>(ser: StateSerialized) -> (Vec<Entry<T>>, Option<PlotBounds>) {
	let mut result = Vec::new();
	let bounds = ser.default_bounds.map(|b| PlotBounds::from_min_max([b[0], b[1]], [b[2], b[3]]));
	for (id, entry) in ser.entries.into_iter().enumerate() {
		let entry_deserialized = Entry {
			id:      id as u64,
			name:    entry.name,
			visible: entry.visible,
			color:   entry.color,
			ty:      match entry.ty {
				EntryTypeSerialized::Function {
					func,
					ranged,
					range_start,
					range_end,
					style,
					implicit_resolution,
				} => EntryType::Function {
					func: func.into_expr(),
					// Actual type will be set in compilation step later
					ty: if ranged { FunctionType::Ranged } else { FunctionType::X },
					range_start: range_start.into_expr(),
					range_end: range_end.into_expr(),
					implicit_resolution: implicit_resolution
						.clamp(MIN_IMPLICIT_RESOLUTION, MAX_IMPLICIT_RESOLUTION),

					style,
				},
				EntryTypeSerialized::Constant { value, step, ty } => {
					EntryType::Constant { value: T::Float::f64_to_float(value), step, ty }
				},
				EntryTypeSerialized::Points(points) => {
					let mut points_deserialized = Vec::new();
					for point in points {
						let point_deserialized = PointEntry {
							x:               point.x.into_expr(),
							y:               point.y.into_expr(),
							drag_point: None,
              drag_type: point.drag_type,
              both_drag_dirs_available: true,
						};
						points_deserialized.push(point_deserialized);
					}
					EntryType::Points(points_deserialized)
				},
				EntryTypeSerialized::Integral { func: f, lower: l, upper: u, resolution: r } => {
					EntryType::Integral {
						func:  f.into_expr(),
						lower: l.into_expr(),
						upper: u.into_expr(),

						calculated: None,
						resolution: r.max(10),
					}
				},
				EntryTypeSerialized::Label { x: text_x, y: text_y, size, underline } => EntryType::Label {
					x: text_x.into_expr(),
					y: text_y.into_expr(),
					size: size.into_expr(),
					underline,
				},
			},
		};
		result.push(entry_deserialized);
	}
	(result, bounds)
}
pub fn deserialize_from_json<T: EvalexprNumericTypes>(
	reader: &[u8],
) -> Result<(Vec<Entry<T>>, Option<PlotBounds>), String> {
	let entries: StateSerialized = serde_json::from_slice(reader).map_err(|e| e.to_string())?;
	Ok(entries_from_ser(entries))
}
// pub fn deserialize_from_url<T: EvalexprNumericTypes>(url: &str) -> Result<Vec<Entry<T>>, String> {
// }
#[cfg(target_arch = "wasm32")]
pub fn save_file<T: EvalexprNumericTypes>(
	ui_state: &mut UiState, state: &State<T>, frame: &mut eframe::Frame,
) {
	let file = format!("{}.json", state.name);
	let mut output = Vec::new();
	if let Err(e) = serialize_to_json(&mut output, &state.entries, Some(&ui_state.plot_bounds)) {
		ui_state.serialization_error = Some(e.to_string());
	} else {
		ui_state.serialization_error = None;
		ui_state.web_storage.insert(file, String::from_utf8(output).unwrap());
		if let Some(storage) = frame.storage_mut() {
			storage.flush();
		}
	}
}
#[cfg(not(target_arch = "wasm32"))]
pub fn save_file<T: EvalexprNumericTypes>(
	ui_state: &mut UiState, state: &State<T>, _frame: &mut eframe::Frame,
) {
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
	if let Err(e) = serialize_to_json(&mut file, &state.entries, Some(&ui_state.plot_bounds) ){
		ui_state.serialization_error = Some(e.to_string());
	} else {
		ui_state.serialization_error = None;
		load_file_entries(&ui_state.cur_dir, &mut ui_state.web_storage);
	}
}
pub fn load_file_entries(cur_dir: &str, web_st: &mut FxHashMap<String, String>) {
	web_st.clear();
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

		web_st.insert(file_name.to_string(), String::new());
	}
}
#[cfg(target_arch = "wasm32")]
pub fn load_file<T: EvalexprNumericTypes>(
	cur_dir: &str, web_st: &FxHashMap<String, String>, file_name: &str, state: &mut State<T>,
) -> Result<(), String> {
	if let Some(file) = web_st.get(file_name) {
		let (entries,bounds) = deserialize_from_json::<T>(file.as_bytes())?;
		state.entries = entries;
    state.default_bounds = bounds;

		state.name = file_name.strip_suffix(".json").unwrap_or(file_name).to_string();
		state.clear_cache = true;
	}
	Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn load_file<T: EvalexprNumericTypes>(
	cur_dir: &str, _web_st: &FxHashMap<String, String>, file_name: &str, state: &mut State<T>,
) -> Result<(), String> {
	let Ok(file) = std::fs::read(PathBuf::from(cur_dir).join(file_name)) else {
		return Err(format!("Could not open file: {}", file_name));
	};
	let (entries, default_bounds) =
		deserialize_from_json::<T>(&file).map_err(|e| format!("Could not deserialize file: {}", e))?;
	state.entries = entries;
	state.default_bounds = default_bounds;

	state.name = file_name.strip_suffix(".json").unwrap_or(file_name).to_string();
	state.clear_cache = true;

	Ok(())
}

pub fn persistence_ui<T: EvalexprNumericTypes>(
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
				load_file_entries(&ui_state.cur_dir, &mut ui_state.web_storage);
			}
		});
	}

	if !ui_state.web_storage.is_empty() {
		ui.separator();
	}
	ui.horizontal(|ui| {
		Grid::new("files").num_columns(3).striped(true).show(ui, |ui| {
			for file_name in ui_state.web_storage.keys() {
				ui.label(file_name);
				if ui.button("Load").clicked() {
					if let Err(e) = load_file(&ui_state.cur_dir, &ui_state.web_storage, file_name, state) {
						ui_state.serialization_error = Some(format!("Could not open file: {}", e));
						return;
					};
					ui_state.serialization_error = None;
					ui_state.next_id += state.entries.len() as u64;
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
							ui_state.web_storage.remove(file);
						}
						#[cfg(not(target_arch = "wasm32"))]
						{
							if let Ok(()) = std::fs::remove_file(PathBuf::from(&ui_state.cur_dir).join(file)) {
								ui_state.web_storage.remove(file);
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
