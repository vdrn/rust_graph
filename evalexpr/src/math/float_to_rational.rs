use crate::math::gcd;

pub fn decimal_places(v: f64, sig_figs:u32) -> u32 {
	let magnitude = v.abs().log10().floor();
	 ((sig_figs as f64 - 1.0 - magnitude).max(0.0) as u32).min(sig_figs)
}
pub fn f64_to_rational_display(v: f64, sig_figs: u32) -> (i64, u64) {
	if v == 0.0 {
		return (0, 1);
	}

	let sign = v.signum() as i64;
	let v = v.abs();

  let decimal_places = decimal_places(v, sig_figs);

	// Format with calculated precision
	let s = format!("{:.prec$}", v, prec = decimal_places as usize);

	// Count actual decimal places (trimming trailing zeros)
	let decimal_places = s.split('.').nth(1).map(|d| d.trim_end_matches('0').len()).unwrap_or(0);

	if decimal_places == 0 {
		return (sign * v.round() as i64, 1);
	}

	let multiplier = 10_u64.pow(decimal_places as u32);
	let numerator = (v * multiplier as f64).round() as i64;

	let g = gcd::gcd(numerator.unsigned_abs(), multiplier);
	(sign * (numerator / g as i64), multiplier / g)
}
