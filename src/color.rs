use eframe::egui::Color32;
use evalexpr::{EvalexprFloat, ExpressionFunction, HashMapContext, Stack, Value};
use serde::{Deserialize, Serialize};

use crate::entry::ColorEntryType;


pub struct ProcessedColors<T: EvalexprFloat> {
	pub colors: Vec<(u64, ProcessedColor<T>)>,
}
impl<T: EvalexprFloat> ProcessedColors<T> {
	pub fn new() -> Self { Self { colors: Vec::new() } }
	pub fn add_constant(&mut self, id: u64, name: String, color: &Value<T>, ty: ColorEntryType) {
		let color = value_to_color(color, ty).unwrap_or(Color32::TRANSPARENT);
		self.colors.push((id, ProcessedColor { color: ProcessedColorExpr::Constant(color), name }));
	}
	pub fn add_function(&mut self, id: u64, name: String, func: ExpressionFunction<T>, ty: ColorEntryType) {
		self.colors.push((id, ProcessedColor { color: ProcessedColorExpr::Function(func, ty), name }));
	}
	pub fn clear(&mut self) { self.colors.clear(); }
	pub fn sort(&mut self) { self.colors.sort_unstable_by_key(|(id, _)| *id); }
	pub fn find_color(&self, id: u64) -> Option<&ProcessedColor<T>> {
		self.colors.iter().find(|(i, _)| *i == id).map(|(_, c)| c)
	}
}
#[derive(Clone)]
pub struct ProcessedColor<T: EvalexprFloat> {
	pub color: ProcessedColorExpr<T>,
	pub name:  String,
}

#[derive(Clone)]
pub enum ProcessedColorExpr<T: EvalexprFloat> {
	Constant(Color32),
	Function(ExpressionFunction<T>, ColorEntryType),
}
impl<T: EvalexprFloat> ProcessedColor<T> {
	pub fn get_color32(
		&self, stack: &mut Stack<T>, ctx: &HashMapContext<T>, x: T, y: T, v: T,
	) -> Result<Color32, String> {
		match &self.color {
			ProcessedColorExpr::Constant(color) => Ok(*color),
			ProcessedColorExpr::Function(func, ty) => value_to_color(
				&func
					.call(stack, ctx, &[Value::Float(x), Value::Float(y), Value::Float(v)])
					.map_err(|e| e.to_string())?,
				*ty,
			),
		}
	}
}

pub fn value_to_color<T: EvalexprFloat>(value: &Value<T>, ty: ColorEntryType) -> Result<Color32, String> {
	match value {
		Value::Tuple(thin_vec) => match thin_vec.len() {
			3 | 4 => {
				let r = thin_vec[0].as_float().map_err(|e| e.to_string())?.to_f64();
				let g = thin_vec[1].as_float().map_err(|e| e.to_string())?.to_f64();
				let b = thin_vec[2].as_float().map_err(|e| e.to_string())?.to_f64();
				let a = if thin_vec.len() == 4 {
					Some(thin_vec[3].as_float().map_err(|e| e.to_string())?.to_f64())
				} else {
					None
				};
				return Ok(match ty {
					ColorEntryType::Rgb => {
						Color32::from_rgba_unmultiplied(r as u8, g as u8, b as u8, a.unwrap_or(255.0) as u8)
					},
					ColorEntryType::RgbNormalized => Color32::from_rgba_unmultiplied(
						(r * 255.0) as u8,
						(g * 255.0) as u8,
						(b * 255.0) as u8,
						(a.unwrap_or(1.0) * 255.0) as u8,
					),
					ColorEntryType::Hsl => hsl_to_rgb(
						r as f32 / 360.0,
						g as f32 / 255.0,
						b as f32 / 255.0,
						(a).unwrap_or(255.0) as f32 / 255.0,
					),
					ColorEntryType::HslNormalized => {
						hsl_to_rgb(r as f32, g as f32, b as f32, a.unwrap_or(1.0) as f32)
					},
					ColorEntryType::Oklaba => {
						oklaba_to_rgb(r as f32, g as f32, b as f32, (a).unwrap_or(1.0) as f32)
					},
				});
			},
			_ => {},
		},
		_ => {},
	}
	Err("Expected a 3 or 4 element tuple".to_string())
}

#[rustfmt::skip]
pub fn hsl_to_rgb(h: f32, s: f32, l: f32, a:f32) -> Color32 {
    let r;
    let g;
    let b;

    if s == 0.0 {  r = l; g = l; b = l; }
    else {
        fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
            if t < 0.0 { t += 1.0 }
            if t > 1.0 { t -= 1.0 }
            if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
            if t < 1.0 / 2.0 { return q; }
            if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
            p
        }

        let q = if l < 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let p = 2.0 * l - q;
        r = hue_to_rgb(p, q, h + 1.0 / 3.0);
        g = hue_to_rgb(p, q, h);
        b = hue_to_rgb(p, q, h - 1.0 / 3.0);
    }

    Color32::from_rgba_unmultiplied((r*255.0) as u8, (g*255.0) as u8, (b*255.0) as u8, (a*255.0) as u8)
}

/// Arguments:
///
/// * `l`: Perceived lightness
/// * `a`: How green/red the color is
/// * `b`: How blue/yellow the color is
/// * `alpha`: Alpha [0..1]
#[allow(clippy::excessive_precision)]
#[inline]
pub fn oklaba_to_rgb(l: f32, a: f32, b: f32, alpha: f32) -> Color32 {
	let l_ = (l + 0.3963377774 * a + 0.2158037573 * b).powi(3);
	let m_ = (l - 0.1055613458 * a - 0.0638541728 * b).powi(3);
	let s_ = (l - 0.0894841775 * a - 1.2914855480 * b).powi(3);

	let r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_;
	let g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_;
	let b = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_;

	linear_rgb_to_rgb(r, g, b, alpha)
}

#[inline]
pub fn linear_rgb_to_rgb(r: f32, g: f32, b: f32, a: f32) -> Color32 {
	fn from_linear(x: f32) -> f32 {
		if x >= 0.0031308 {
			return 1.055 * x.powf(1.0 / 2.4) - 0.055;
		}
		12.92 * x
	}
	Color32::from_rgba_unmultiplied(
		(from_linear(r) * 255.) as u8,
		(from_linear(g) * 255.) as u8,
		(from_linear(b) * 255.) as u8,
		(a * 255.) as u8,
	)
}

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
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum EntryColor {
	DefaultColor(usize),
	CustomColor(u64),
}

impl Default for EntryColor {
	fn default() -> Self { Self::DefaultColor(0) }
}

impl EntryColor {
	pub fn get_base_color<T: EvalexprFloat>(
		&self, processed_colors: &ProcessedColors<T>, ctx: &evalexpr::HashMapContext<T>, stack: &mut Stack<T>,
	) -> Result<Color32, String> {
		match self {
			EntryColor::DefaultColor(i) => Ok(COLORS[*i % COLORS.len()]),
			EntryColor::CustomColor(id) => {
				if let Some(color) = processed_colors.find_color(*id) {
					Ok(color.get_color32(stack, ctx, T::ZERO, T::ZERO, T::ZERO)?)
				} else {
					Err(format!("Color with id {} not found", id))
				}
			},
		}
	}
}
