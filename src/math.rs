use egui_plot::PlotPoint;

pub fn zoom_in_x_on_nan_boundary(
	a: (f64, f64), b: (f64, f64), eps: f64, mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<(f64, f64)> {
	// If both are nans or both are defined, no need to do anything
	if a.1.is_nan() == b.1.is_nan() {
		return None;
	}

	let mut left = a;
	let mut right = b;

	let mut prev_mid_x = None;
	while (right.0 - left.0).abs() > eps {
		let mid_x = (left.0 + right.0) * 0.5;
		let mid_y = eval(mid_x)?;
		if prev_mid_x == Some(mid_x) {
			break;
		}
		prev_mid_x = Some(mid_x);

		let mid = (mid_x, mid_y);

		if left.1.is_nan() == mid_y.is_nan() {
			left = mid;
		} else {
			right = mid;
		}
	}

	Some(if left.1.is_nan() { right } else { left })
}

pub struct DiscontinuityDetector {
	prev_value:      Option<(f64, f64)>,
	prev_abs_change: f64,
	eps:             f64,
}
impl DiscontinuityDetector {
	pub fn new_with_initial(step_size: f64, eps: f64, initial_value: (f64, f64)) -> Self {
		Self {
			prev_value:      Some(initial_value),
			prev_abs_change: 0.0,
			eps:             (step_size * 0.01).max(eps),
		}
	}
	pub fn new(step_size: f64, eps: f64) -> Self {
		Self { prev_value: None, prev_abs_change: 0.0, eps: (step_size * 0.01).max(eps) }
	}
	pub fn detect(
		&mut self, arg: f64, value: f64, mut eval: impl FnMut(f64) -> Option<f64>,
	) -> Option<((f64, f64), (f64, f64))> {
		let prev_value = self.prev_value.replace((arg, value));
		if value.is_nan() {
			return None;
		}

		if let Some((prev_arg, prev_value)) = prev_value {
			if prev_value.is_nan() {
				return None;
			}
			let abs_change = (value - prev_value).abs();

			// println!("abs_change: {abs_change}");
			if abs_change > self.prev_abs_change * 2.0 {
				let midpoint = (prev_arg + arg) * 0.5;
				let value_at_midpoint = eval(midpoint)?;

				let left_change = (value_at_midpoint - prev_value).abs();
				let right_change = (value - value_at_midpoint).abs();

				let max_half = left_change.max(right_change);
				if max_half > 0.8 * abs_change {
					// let min_half = left_change.min(right_change);
					// let ratio = min_half / max_half;
					// if ratio < 0.2 {
					// println!("max_half {max_half} abs_change {abs_change}");
					// discontinuity detected - find exact points via bisection
					return Some(self.bisect_discontinuity(prev_arg, prev_value, arg, value, &mut eval));
				}
			}

			self.prev_abs_change = abs_change;
		}

		None
	}

	fn bisect_discontinuity(
		&self, mut left_arg: f64, mut left_value: f64, mut right_arg: f64, mut right_value: f64,
		eval: &mut impl FnMut(f64) -> Option<f64>,
	) -> ((f64, f64), (f64, f64)) {
		// Bisect until we're within eps
		while (right_arg - left_arg) > self.eps {
			let mid_arg = (left_arg + right_arg) * 0.5;
			let Some(mid_value) = eval(mid_arg) else {
				break;
			};

			let left_change = (mid_value - left_value).abs();
			let right_change = (right_value - mid_value).abs();

			// Move toward the side with the larger jump
			if left_change < right_change {
				left_arg = mid_arg;
				left_value = mid_value;
			} else {
				right_arg = mid_arg;
				right_value = mid_value;
			}
		}

		((left_arg, left_value), (right_arg, right_value))
	}
}

pub fn closest_point_on_segment(a: (f64, f64), b: (f64, f64), point: (f64, f64)) -> (f64, f64) {
	let (ax, ay) = a;
	let (bx, by) = b;
	let (px, py) = point;

	let ab_x = bx - ax;
	let ab_y = by - ay;

	let ap_x = px - ax;
	let ap_y = py - ay;

	let ab_len_sq = ab_x * ab_x + ab_y * ab_y;

	if ab_len_sq == 0.0 {
		// a == b
		return a;
	}

	let t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;

	let t = t.clamp(0.0, 1.0);

	(ax + t * ab_x, ay + t * ab_y)
}
pub fn dist_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
	let (ax, ay) = a;
	let (bx, by) = b;
	let a = ax - bx;
	let b = ay - by;
	a * a + b * b
}

pub fn solve_secant(
	cur_x: f64, cur_y: f64, target_y: f64, eps: f64, mut f: impl FnMut(f64) -> f64,
) -> Option<f64> {
	// println!("secant cur_x: {cur_x} cur_y: {cur_y} target_y: {target_y}");
	// find root f(x) - target_y = 0
	let max_iterations = 40;
	let h = 0.0001; // small step for second point

	// initial points
	let mut x0 = cur_x;
	let mut y0 = cur_y - target_y;

	let mut x1 = cur_x + h;
	let mut y1 = f(x1) - target_y;

	for _ in 0..max_iterations {
		if y1.abs() < eps {
			// success
			return Some(x1);
		}

		if (y1 - y0) == 0.0 {
			// div by zero
			// We do best effort guess here.
			// TODO: this is not the solution!
			// The real solution is to disallow dragging of non-injective expressions.
			if (x0 - cur_x).abs() < (x1 - cur_x).abs() {
				// x0 is closer to cur_x
				return Some(x0);
			}
			// x1 is closer to cur_x
			return Some(x1);
		}

		let x_next = x1 - y1 * (x1 - x0) / (y1 - y0);
		if !x_next.is_finite() {
			// nan or inf
			// println!("nan or inf");
			return None;
		}

		x0 = x1;
		y0 = y1;
		x1 = x_next;
		y1 = f(x1) - target_y;
	}

	if y1.abs() < eps {
		Some(x1)
	} else {
		// failed to converge
		// println!("failed to converge x0: {x0} x1: {x1} y0: {y0} y1: {y1}");
		None
	}
}
pub fn intersect_segs(
	a1: PlotPoint, a2: PlotPoint, b1: PlotPoint, b2: PlotPoint, eps: f64,
) -> Option<PlotPoint> {
	let d1x = a2.x - a1.x;
	let d1y = a2.y - a1.y;
	let d2x = b2.x - b1.x;
	let d2y = b2.y - b1.y;

	let denom = d1x * d2y - d1y * d2x;

	if denom.abs() < eps {
		// parallel
		return None;
	}

	let dx = b1.x - a1.x;
	let dy = b1.y - a1.y;

	let t = (dx * d2y - dy * d2x) / denom;
	let u = (dx * d1y - dy * d1x) / denom;

	if t >= -eps && t <= 1.0 + eps && u >= -eps && u <= 1.0 + eps {
		Some(PlotPoint { x: a1.x + t * d1x, y: a1.y + t * d1y })
	} else {
		None
	}
}

pub fn minimize(cur_value: f64, mouse_pos: (f64, f64), eps: f64, mut f: impl FnMut(f64) -> (f64, f64)) -> f64 {
	// Levenberg-Marquardt
	// minimizing abs(f(val) - mouse_pos)^2

	let mut t = cur_value;
	let max_iterations = 100;

	for _ in 0..max_iterations {
		let (fx, fy) = f(t);
		let residual_x = fx - mouse_pos.0;
		let residual_y = fy - mouse_pos.1;

		let error = residual_x * residual_x + residual_y * residual_y;

		if error < eps {
			// success
			// println!("succes at t: {t}");
			return t;
		}

		let h = eps * 100.0;
		let (fx_h, fy_h) = f(t + h);

		// first derivative
		let df_dt_x = (fx_h - fx) / h;
		let df_dt_y = (fy_h - fy) / h;

		let gradient = 2.0 * (residual_x * df_dt_x + residual_y * df_dt_y);

		// second derivative (hessian)
		let (fx_2h, fy_2h) = f(t + 2.0 * h);
		let d2f_dt2_x = (fx_2h - 2.0 * fx_h + fx) / (h * h);
		let d2f_dt2_y = (fy_2h - 2.0 * fy_h + fy) / (h * h);
		// Hessian of dist sq
		let hessian =
			2.0 * (df_dt_x * df_dt_x + df_dt_y * df_dt_y + residual_x * d2f_dt2_x + residual_y * d2f_dt2_y);

		let damping = eps * 1000.0;
		let step = -gradient / (hessian.abs() + damping);

		// line search
		let mut best_t = t;

		for scale in [1.0, 0.5, 0.25, 0.1] {
			let t_new = t + scale * step;
			let (fx_new, fy_new) = f(t_new);
			let rx_new = fx_new - mouse_pos.0;
			let ry_new = fy_new - mouse_pos.1;
			let error_new = rx_new * rx_new + ry_new * ry_new;

			if error_new < error {
				best_t = t_new;
				break;
			}
		}

		if (best_t - t).abs() < eps {
			// we're stuck
			return t;
		}

		t = best_t;
	}

	// println!("reached max iterations {t}");
	t
}

pub fn newton_raphson_minimizer2<const ITERS: usize>(
	start_x: f64, start_y: f64, eps: f64, mut f: impl FnMut(f64, f64) -> Result<f64, String>,
) -> Result<Option<(f64, f64)>, String> {
	let mut x = start_x;
	let mut y = start_y;

	for _ in 0..ITERS {
		let f_val = f(x, y)?;
		let df_dx = (f(x + eps, y)? - f(x - eps, y)?) / (2.0 * eps);
		let df_dy = (f(x, y + eps)? - f(x, y - eps)?) / (2.0 * eps);
		// let df_dx = (f(x + eps, y)? - f_val) / (eps);
		// let df_dy = (f(x, y + eps)? - f_val) / (eps);

		let grad_len_sq = df_dx * df_dx + df_dy * df_dy;

		// gradient too small
		if grad_len_sq < 1e-10 {
			return Ok(None);
		}

		let step = f_val / grad_len_sq;
		x -= step * df_dx;
		y -= step * df_dy;
	}

	Ok(Some((x, y)))
}

pub fn newton_raphson_minimizer2_tmp<const ITERS: usize>(
	start_x: f64, start_y: f64, eps: f64, mut f: impl FnMut(f64, f64) -> Result<f64, String>,
) -> Result<Option<(f64, f64)>, String> {
	let mut x = start_x;
	let mut y = start_y;

	let initial_f_val = f(x, y)?;
	let initial_f = initial_f_val.abs();
	let mut f_val = initial_f_val;

	let mut best_f = initial_f;
	let mut best_x = x;
	let mut best_y = y;

	for _ in 0..ITERS {
		let df_dx = (f(x + eps, y)? - f(x - eps, y)?) / (2.0 * eps);
		let df_dy = (f(x, y + eps)? - f(x, y - eps)?) / (2.0 * eps);

		let grad_len_sq = df_dx * df_dx + df_dy * df_dy;

		// gradient too small
		if grad_len_sq < 1e-10 {
			break;
		}

		// move in the direction of the gradient
		let step = f_val / grad_len_sq;
		let dx = -step * df_dx;
		let dy = -step * df_dy;

		// estimate new f
		f_val = f_val + df_dx * dx + df_dy * dy;
		let estimated_new_f = f_val.abs();

		// not improving, bail out
		if estimated_new_f >= best_f {
			break;
		}

		x += dx;
		y += dy;
		best_f = estimated_new_f;
		best_x = x;
		best_y = y;
	}

	Ok(if best_f < initial_f { Some((best_x, best_y)) } else { None })
}

pub fn pseudoangle(x: f64, y: f64) -> f64 {
	let p = x / (x.abs() + y.abs()); // -1 .. 1 increasing with x
	// if dy < 0.0 {
	//   3.0 + p //  2 .. 4 increasing with x
	// } else {
	//   1.0 - p //  0 .. 2 decreasing with x
	// }
	-y.signum() * (p + 1.0) + 2.0
}
