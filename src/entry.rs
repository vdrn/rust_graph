use core::ops::RangeInclusive;

use eframe::egui::Color32;
use evalexpr::{EvalexprNumericTypes,EvalexprFloat, Node};
use serde::{Deserialize, Serialize};

pub const COLORS: &'static [Color32; 20] = &[
	Color32::from_rgb(255, 107, 107), // Bright coral red
	Color32::from_rgb(78, 205, 196),  // Turquoise
	Color32::from_rgb(69, 183, 209),  // Sky blue
	Color32::from_rgb(255, 160, 122), // Light salmon
	Color32::from_rgb(152, 216, 200), // Mint green
	Color32::from_rgb(255, 217, 61),  // Golden yellow
	Color32::from_rgb(107, 207, 127), // Fresh green
	Color32::from_rgb(199, 125, 255), // Bright purple
	Color32::from_rgb(255, 133, 161), // Pink
	Color32::from_rgb(93, 173, 226),  // Bright blue
	Color32::from_rgb(248, 183, 57),  // Orange
	Color32::from_rgb(127, 219, 255), // Aqua
	Color32::from_rgb(57, 255, 20),   // Neon green
	Color32::from_rgb(255, 20, 147),  // Deep pink
	Color32::from_rgb(0, 217, 255),   // Cyan
	Color32::from_rgb(255, 179, 71),  // Peach
	Color32::from_rgb(139, 92, 246),  // Violet
	Color32::from_rgb(52, 211, 153),  // Emerald
	Color32::from_rgb(244, 114, 182), // Hot pink
	Color32::from_rgb(251, 191, 36),  // Amber
];
pub const NUM_COLORS: usize = COLORS.len();

#[derive(Clone, Debug)]
pub struct Entry<T: EvalexprNumericTypes> {
	pub name:    String,
	pub visible: bool,
	pub color:   usize,
	pub value:   EntryData<T>,
}
impl<T: EvalexprNumericTypes> Entry<T> {
	pub fn symbol(&self) -> &'static str {
		match self.value {
			EntryData::Function { .. } => "λ",
			EntryData::Constant { .. } => {
				if self.visible {
					"⏸"
				} else {
					"⏵"
				}
			},
			EntryData::Points(_) => "●",
			EntryData::Integral { .. } => "∫",
		}
	}
}
#[derive(Clone, Debug)]
pub struct EntryPoint<T: EvalexprNumericTypes> {
	pub text_x: String,
	pub x:      Option<Node<T>>,
	pub text_y: String,
	pub y:      Option<Node<T>>,
}

impl<T: EvalexprNumericTypes> Default for EntryPoint<T> {
	fn default() -> Self {
		Self {
			text_x: Default::default(),
			x:      Default::default(),
			text_y: Default::default(),
			y:      Default::default(),
		}
	}
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstantType {
	LoopForwardAndBackward { start: f64, end: f64, forward: bool },
	LoopForward { start: f64, end: f64 },
	PlayOnce { start: f64, end: f64 },
	PlayIndefinitely { start: f64 },
}
impl ConstantType {
	pub fn range(&self) -> RangeInclusive<f64> {
		match self {
			ConstantType::LoopForwardAndBackward { start, end, .. } => (*start..=*end).into(),
			ConstantType::LoopForward { start, end } => (*start..=*end).into(),
			ConstantType::PlayOnce { start, end } => (*start..=*end).into(),
			ConstantType::PlayIndefinitely { start } => (*start..=f64::INFINITY).into(),
		}
	}
	pub fn symbol(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁",
			ConstantType::LoopForward { .. } => "🔂",
			ConstantType::PlayOnce { .. } => "⏯",
			ConstantType::PlayIndefinitely { .. } => "🔀",
		}
	}
	pub fn name(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁 Loop forward and Backward",
			ConstantType::LoopForward { .. } => "🔂 Loop forward",
			ConstantType::PlayOnce { .. } => "⏯ Play once",
			ConstantType::PlayIndefinitely { .. } => "🔀 Play indefinitely",
		}
	}
}
#[derive(Clone, Debug)]
pub enum EntryData<T: EvalexprNumericTypes> {
	Function {
		text: String,
		func: Option<Node<T>>,
	},
	Constant {
		value: T::Float,
		step:  f64,
		ty:    ConstantType,
	},
	Points(Vec<EntryPoint<T>>),
	Integral {
		func_text:  String,
		func:       Option<Node<T>>,
		lower_text: String,
		lower:      Option<Node<T>>,
		upper_text: String,
		upper:      Option<Node<T>>,
		calculated: Option<T::Float>,
		resolution: usize,
	},
}
#[derive(Clone, Debug, PartialEq, Copy)]
pub enum EntryType {
	Function,
	Constant,
	Points,
	Integral,
}

impl<T: EvalexprNumericTypes> Entry<T> {
	pub fn color(&self) -> Color32 { COLORS[self.color % NUM_COLORS] }
	pub fn new_function(color: usize, text: String) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			value:   EntryData::Function { text, func: None },
		}
	}
	pub fn new_constant(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: false,
			name:    String::new(),
			value:   EntryData::Constant {
				value: T::Float::ZERO,
				step:  0.01,
				ty:    ConstantType::LoopForwardAndBackward { start: 0.0, end: 2.0, forward: true },
			},
		}
	}
	pub fn new_points(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			value:   EntryData::Points(vec![EntryPoint::default()]),
		}
	}
	pub fn new_integral(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			value:   EntryData::Integral {
				func_text:  String::new(),
				func:       None,
				lower_text: String::new(),
				lower:      None,
				upper_text: String::new(),
				upper:      None,
				calculated: None,
				resolution: 500,
			},
		}
	}
}
