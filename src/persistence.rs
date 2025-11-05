use std::io::Write;
use std::path::PathBuf;

use ahash::AHashMap;
use evalexpr::{EvalexprFloat, EvalexprNumericTypes};
use serde::{Deserialize, Serialize};

use crate::{ConstantType, Entry, EntryData, EntryPoint, State, UiState};

#[derive(Serialize, Deserialize)]
pub struct EntrySerialized {
	name:    String,
	visible: bool,
	color:   usize,
	value:   EntryValueSerialized,
}
#[derive(Serialize, Deserialize)]
pub enum EntryValueSerialized {
	Function(String),
	Constant {
		value: f64,
		step:  f64,
		ty:    ConstantType,
	},
	Points(Vec<EntryPointSerialized>),
	Integral {
		func_text:  String,
		lower_text: String,
		upper_text: String,

		#[serde(default)]
		resolution: usize,
	},
}
#[derive(Serialize, Deserialize)]
pub struct EntryPointSerialized {
	x: String,
	y: String,
}

pub fn serialize_to<T: EvalexprNumericTypes>(writer: impl Write, entries: &[Entry<T>]) -> std::io::Result<()> {
	let mut result = Vec::new();
	for entry in entries {
		let entry_serialized = EntrySerialized {
			name:    entry.name.clone(),
			visible: entry.visible,
			color:   entry.color,
			value:   match &entry.value {
				EntryData::Function { text, .. } => EntryValueSerialized::Function(text.clone()),
				EntryData::Constant { value, step, ty } => {
					EntryValueSerialized::Constant { value: value.to_f64(), step: *step, ty: ty.clone() }
				},
				EntryData::Points(points) => {
					let mut points_serialized = Vec::new();
					for point in points {
						let point_serialized =
							EntryPointSerialized { x: point.text_x.clone(), y: point.text_y.clone() };
						points_serialized.push(point_serialized);
					}
					EntryValueSerialized::Points(points_serialized)
				},
				EntryData::Integral { func_text, lower_text, upper_text, resolution, .. } => {
					EntryValueSerialized::Integral {
						func_text:  func_text.clone(),
						lower_text: lower_text.clone(),
						upper_text: upper_text.clone(),
						resolution: *resolution,
					}
				},
			},
		};
		result.push(entry_serialized);
	}
	serde_json::to_writer(writer, &result)?;
	Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn deserialize_from_url<T: EvalexprNumericTypes>() -> Result<Vec<Entry<T>>, String> {
	let href = web_sys::window()
		.expect("Couldn't get window")
		.document()
		.expect("Couldn't get document")
		.location()
		.expect("Couldn't get location")
		.href()
		.expect("Couldn't get href");

	if !href.contains('#') {
		return Ok(Vec::new());
	}
	let Some(without_prefix) = href.split('#').last() else {
		return Ok(Vec::new());
	};

	let decoded = urlencoding::decode(without_prefix).map_err(|e| e.to_string())?;
	deserialize_from(decoded.as_bytes())
}

pub fn deserialize_from<T: EvalexprNumericTypes>(reader: &[u8]) -> Result<Vec<Entry<T>>, String> {
	let entries: Vec<EntrySerialized> = serde_json::from_slice(reader).map_err(|e| e.to_string())?;
	let mut result = Vec::new();
	for entry in entries {
		let entry_deserialized = Entry {
			name:    entry.name,
			visible: entry.visible,
			color:   entry.color,
			value:   match entry.value {
				EntryValueSerialized::Function(text) => {
					EntryData::Function { func: evalexpr::build_operator_tree::<T>(&text).ok(), text }
				},
				EntryValueSerialized::Constant { value, step, ty } => {
					EntryData::Constant { value: T::Float::f64_to_float(value), step, ty }
				},
				EntryValueSerialized::Points(points) => {
					let mut points_deserialized = Vec::new();
					for point in points {
						let point_deserialized = EntryPoint {
							x:      evalexpr::build_operator_tree::<T>(&point.x).ok(),
							y:      evalexpr::build_operator_tree::<T>(&point.y).ok(),
							text_x: point.x,
							text_y: point.y,
						};
						points_deserialized.push(point_deserialized);
					}
					EntryData::Points(points_deserialized)
				},
				EntryValueSerialized::Integral { func_text, lower_text, upper_text, resolution } => {
					EntryData::Integral {
						func: evalexpr::build_operator_tree::<T>(&func_text).ok(),
						lower: evalexpr::build_operator_tree::<T>(&lower_text).ok(),
						upper: evalexpr::build_operator_tree::<T>(&upper_text).ok(),
						func_text,
						lower_text,
						upper_text,
						calculated: None,
						resolution: resolution.max(10),
					}
				},
			},
		};
		result.push(entry_deserialized);
	}
	Ok(result)
}
#[cfg(target_arch = "wasm32")]
pub fn save_file<T: EvalexprNumericTypes>(
	ui_state: &mut UiState, state: &State<T>, frame: &mut eframe::Frame,
) {
	let file = format!("{}.json", state.name);
	let mut output = Vec::new();
	if let Err(e) = serialize_to(&mut output, &state.entries) {
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

		std::fs::create_dir_all(parent).ok();
	}
	let Ok(mut file) = std::fs::File::create(&save_path) else {
		ui_state.serialization_error = Some(format!("Could not create file: {}", save_path.display()));
		return;
	};
	if let Err(e) = serialize_to(&mut file, &state.entries) {
		ui_state.serialization_error = Some(e.to_string());
	} else {
		ui_state.serialization_error = None;
		load_file_entries(&ui_state.cur_dir, &mut ui_state.web_storage);
	}
}
pub fn load_file_entries(cur_dir: &str, web_st: &mut AHashMap<String, String>) {
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
	cur_dir: &str, web_st: &AHashMap<String, String>, file_name: &str, state: &mut State<T>,
) -> Result<(), String> {
	if let Some(file) = web_st.get(file_name) {
		let entries = deserialize_from::<T>(file.as_bytes()).unwrap();
		state.entries = entries;
		state.name = file_name.strip_suffix(".json").unwrap().to_string();
		state.clear_cache = true;
	}
	Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn load_file<T: EvalexprNumericTypes>(
	cur_dir: &str, _web_st: &AHashMap<String, String>, file_name: &str, state: &mut State<T>,
) -> Result<(), String> {
	let Ok(mut file) = std::fs::read(PathBuf::from(cur_dir).join(file_name)) else {
		return Err(format!("Could not open file: {}", file_name));
	};
	let entries =
		deserialize_from::<T>(&mut file).map_err(|e| format!("Could not deserialize file: {}", e))?;
	state.entries = entries;
	state.name = file_name.strip_suffix(".json").unwrap().to_string();
	state.clear_cache = true;

	Ok(())
}
