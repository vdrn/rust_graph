use core::ops::RangeInclusive;

use eframe::egui::{self, Color32};
use evalexpr::{EvalexprFloat, ExpressionFunction, FlatNode, IStr, Value, istr};
use serde::{Deserialize, Serialize};

mod entry_plot_elements;
mod entry_processing;
mod entry_ui;
mod drag_point;

pub use entry_plot_elements::{PlotParams, entry_create_plot_elements};
pub use entry_processing::{optimize_entries, prepare_entry, preprecess_fn};
pub use entry_ui::entry_ui;
pub use drag_point::point_dragging;

pub const DEFAULT_IMPLICIT_RESOLUTION: usize = 200;
pub const MAX_IMPLICIT_RESOLUTION: usize = 500;
pub const MIN_IMPLICIT_RESOLUTION: usize = 10;

pub const COLORS: &[Color32; 20] = &[
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
pub struct Entry<T: EvalexprFloat> {
	pub id:      u64,
	pub name:    String,
	pub visible: bool,
	pub color:   usize,
	pub ty:      EntryType<T>,
}
impl<T: EvalexprFloat> core::hash::Hash for Entry<T> {
	fn hash<H: core::hash::Hasher>(&self, state: &mut H) { self.id.hash(state); }
}

#[derive(Clone, Debug)]
pub struct Expr<T: EvalexprFloat> {
	pub text:             String,
	pub display_rational: bool,
	pub node:             Option<FlatNode<T>>,
	pub inlined_node:     Option<FlatNode<T>>,
	pub expr_function:    Option<ExpressionFunction<T>>,
	pub args:             Vec<IStr>,
}

impl<T: EvalexprFloat> Default for Expr<T> {
	fn default() -> Self {
		Self {
			text:             Default::default(),
			display_rational: false,
			node:             Default::default(),
			inlined_node:     Default::default(),
			args:             Default::default(),
			expr_function:    Default::default(),
		}
	}
}
impl<T: EvalexprFloat> Expr<T> {
	fn computed_const(&self) -> Option<Value<T>> {
		let Some(node) = &self.node else { return None };
		if node.as_constant().is_some() {
			// if unoptimized node is constant, no need to display it
			return None;
		}
		if let Some(func) = &self.expr_function {
			func.as_constant()
		} else if let Some(inlined_node) = &self.inlined_node {
			inlined_node.as_constant()
		} else {
			None
		}
	}
	fn from_text(text: &str) -> Self {
		Self {
			node:             evalexpr::build_operator_tree::<T>(text).ok(),
			display_rational: false,
			inlined_node:     None,
			expr_function:    None,
			text:             text.to_string(),
			args:             Vec::new(),
		}
	}

	fn args_to_string(&self) -> String {
		let mut args_string = String::new();
		for (i, arg) in self.args.iter().enumerate() {
			if i > 0 {
				args_string.push_str(", ");
			}
			args_string.push_str(arg.to_str());
		}
		args_string
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FunctionType {
	// ExpressionX,
	// ExpressionY,
	Expression,
	WithCustomParams,
}
#[derive(Clone, Debug)]
pub enum EntryType<T: EvalexprFloat> {
	Function {
		can_be_drawn: bool,
		identifier:   IStr,
		func:         Expr<T>,
		style:        LineStyleConfig,
		ty:           FunctionType,

		parametric:          bool,
		/// used when parametric is `true`
		range_start:         Expr<T>,
		/// used when parametric is `true`
		range_end:           Expr<T>,
		/// used for functions with x,y params
		implicit_resolution: usize,
	},
	Constant {
		istr_name: IStr,
		value:     T,
		step:      f64,
		ty:        ConstantType,
	},
	Points {
		points: Vec<PointEntry<T>>,
		style:  PointStyle,
	},
	Label {
		x:         Expr<T>,
		y:         Expr<T>,
		size:      Expr<T>,
		underline: bool,
	},
	Folder {
		entries: Vec<Entry<T>>,
	},
}

#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub enum LabelSize {
	#[default]
	Small,
	Medium,
	Large,
}
impl LabelSize {
	pub fn size(&self) -> f32 {
		match self {
			LabelSize::Small => 10.0,
			LabelSize::Medium => 16.0,
			LabelSize::Large => 20.0,
		}
	}
}
#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub enum LabelPosition {
	#[default]
	Bottom,
	Top,
	Left,
	Right,
	TLeft,
	TRight,
	BLeft,
	BRight,
}
impl LabelPosition {
	fn symbol(&self) -> &'static str {
		match self {
			LabelPosition::Top => "⮉",
			LabelPosition::Bottom => "⮋",
			LabelPosition::Left => "⮈",
			LabelPosition::Right => "⮊",
			LabelPosition::TLeft => "⬉",
			LabelPosition::TRight => "⬈",
			LabelPosition::BLeft => "⬋",
			LabelPosition::BRight => "⬊",
		}
	}
	fn dir(&self) -> egui::Vec2 {
		match self {
			LabelPosition::Top => egui::Vec2::new(0.0, 1.0),
			LabelPosition::Bottom => egui::Vec2::new(0.0, -1.0),
			LabelPosition::Left => egui::Vec2::new(-1.0, 0.0),
			LabelPosition::Right => egui::Vec2::new(1.0, 0.0),
			LabelPosition::TLeft => egui::Vec2::new(-0.71, 0.71),
			LabelPosition::TRight => egui::Vec2::new(0.71, 0.71),
			LabelPosition::BLeft => egui::Vec2::new(-0.71, -0.71),
			LabelPosition::BRight => egui::Vec2::new(0.71, -0.71),
		}
	}
}

#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub struct LabelConfig {
	#[serde(default)]
	size:   LabelSize,
	#[serde(default)]
	pos:    LabelPosition,
	#[serde(default)]
	italic: bool,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PointStyle {
	show_lines:   bool,
	#[serde(default)]
	show_arrows:  bool,
	show_points:  bool,
	line_style:   LineStyleConfig,
	#[serde(default)]
	label_config: Option<LabelConfig>,
}

impl Default for PointStyle {
	fn default() -> Self {
		Self {
			show_lines:   true,
			show_points:  true,
			show_arrows:  false,
			label_config: Some(LabelConfig::default()),
			line_style:   LineStyleConfig::default(),
		}
	}
}

#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub enum LineStyleType {
	#[default]
	Solid,
	Dotted,
	Dashed,
}
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct LineStyleConfig {
	line_width:      f32,
	#[serde(default)]
	line_style:      LineStyleType,
	line_style_size: f32,
}
impl Default for LineStyleConfig {
	fn default() -> Self { Self { line_width: 1.5, line_style: LineStyleType::Solid, line_style_size: 5.5 } }
}
impl LineStyleConfig {
	fn egui_line_style(&self) -> egui_plot::LineStyle {
		match self.line_style {
			LineStyleType::Solid => egui_plot::LineStyle::Solid,
			LineStyleType::Dotted => egui_plot::LineStyle::Dotted { spacing: self.line_style_size },
			LineStyleType::Dashed => egui_plot::LineStyle::Dashed { length: self.line_style_size },
		}
	}
}

impl<T: EvalexprFloat> Entry<T> {
	pub fn symbol(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "λ",
			EntryType::Constant { .. } => {
				if self.visible {
					"⏸"
				} else {
					"⏵"
				}
			},
			EntryType::Points { .. } => "◊",
			EntryType::Label { .. } => "📃",
			EntryType::Folder { .. } => "📂",
		}
	}
	pub fn type_name(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "λ   Function",
			EntryType::Constant { .. } => "⏵ Constant",
			EntryType::Points { .. } => "◊ Points",
			EntryType::Label { .. } => "📃 Label",
			EntryType::Folder { .. } => "📂 Folder",
		}
	}
	pub fn color(&self) -> Color32 { COLORS[self.color % NUM_COLORS] }
	pub fn new_function(id: u64, text: String) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Function {
				identifier:   istr(""),
				can_be_drawn: true,

				func:                Expr {
					node: evalexpr::build_operator_tree::<T>(&text).ok(),
					display_rational: false,
					inlined_node: None,
					expr_function: None,
					text,
					args: Vec::new(),
				},
				parametric:          false,
				range_start:         Expr::from_text("-2"),
				range_end:           Expr::from_text("2"),
				ty:                  FunctionType::Expression,
				style:               LineStyleConfig::default(),
				implicit_resolution: DEFAULT_IMPLICIT_RESOLUTION,
			},
		}
	}
	pub fn new_constant(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: false,
			name: String::new(),
			ty: EntryType::Constant {
				istr_name: istr(""),
				value:     T::ZERO,
				step:      0.01,
				ty:        ConstantType::LoopForwardAndBackward {
					start:   -10.0,
					end:     10.0,
					forward: true,
				},
			},
		}
	}
	pub fn new_points(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Points { points: vec![PointEntry::default()], style: PointStyle::default() },
		}
	}
	pub fn new_label(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Label {
				x:         Expr::default(),
				y:         Expr::default(),
				size:      Expr::default(),
				underline: false,
			},
		}
	}
	pub fn new_folder(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Folder { entries: Vec::new() },
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum PointDragType {
	#[default]
	Both,
	X,
	Y,
	NoDrag,
}
impl PointDragType {
	pub fn symbol(self) -> &'static str {
		match self {
			PointDragType::Both => "↗",
			PointDragType::X => "↔",
			PointDragType::Y => "↕",
			PointDragType::NoDrag => "○",
		}
	}
	fn name(self) -> &'static str {
		match self {
			PointDragType::Both => "↗ Both",
			PointDragType::X => "↔ X",
			PointDragType::Y => "↕ Y",
			PointDragType::NoDrag => "○ No Drag",
		}
	}
}
#[derive(Clone, Debug)]
pub struct PointEntry<T: EvalexprFloat> {
	pub x:                        Expr<T>,
	pub y:                        Expr<T>,
	pub drag_point:               Option<DragPoint>,
	pub drag_type:                PointDragType,
	pub both_drag_dirs_available: bool,
	pub val:                      Option<(T, T)>,
}

impl<T: EvalexprFloat> Default for PointEntry<T> {
	fn default() -> Self {
		Self {
			x:                        Expr::default(),
			y:                        Expr::default(),
			drag_point:               None,
			drag_type:                PointDragType::default(),
			both_drag_dirs_available: true,
			val:                      None,
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
			ConstantType::LoopForwardAndBackward { start, end, .. } => *start..=*end,
			ConstantType::LoopForward { start, end } => *start..=*end,
			ConstantType::PlayOnce { start, end } => *start..=*end,
			ConstantType::PlayIndefinitely { start } => *start..=f64::INFINITY,
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
			ConstantType::PlayOnce { .. } => "⏭ Play once",
			ConstantType::PlayIndefinitely { .. } => "🔀 Play indefinitely",
		}
	}
}

pub static RESERVED_NAMES: [&str; 2] = ["x", "y"];

pub fn f64_to_value<T: EvalexprFloat>(x: f64) -> Value<T> { Value::<T>::Float(T::f64_to_float(x)) }
pub fn f64_to_float<T: EvalexprFloat>(x: f64) -> T { T::f64_to_float(x) }

#[derive(Clone, Debug)]
pub enum DragPoint {
	BothCoordLiterals,
	XLiteral,
	YLiteral,
	XConstant(IStr),
	YConstant(IStr),
	XLiteralYConstant(IStr),
	YLiteralXConstant(IStr),
	BothCoordConstants(IStr, IStr),
	SameConstantBothCoords(IStr),
}
