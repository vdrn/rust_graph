#[allow(clippy::assign_op_pattern)]
pub fn gcd(mut u: u64, mut v: u64) -> u64 {
	if u == 0 {
		return v;
	}
	if v == 0 {
		return u;
	}

	let shift = (u | v).trailing_zeros();
	u = u >> shift;
	v = v >> shift;
	u = u >> u.trailing_zeros();

	loop {
		v >>= v.trailing_zeros();

		if u > v {
			core::mem::swap(&mut u, &mut v);
		}
		v -= u;

		if v == 0 {
			break;
		}
	}

	u << shift
}
