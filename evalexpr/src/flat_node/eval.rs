use thin_vec::ThinVec;

use crate::{EvalexprError, EvalexprFloat, EvalexprResult, Value};

#[inline(always)]
pub fn eval_range<F: EvalexprFloat>(
    start: Value<F>,
    end: Value<F>,
) -> EvalexprResult<ThinVec<Value<F>>, F> {
    let end = end.as_float()?;
    let (start, second) = match start {
        Value::Float(start) => (start, None),
        Value::Float2(start, second) => (start, Some(second)),
        _ => return Err(EvalexprError::expected_float(start)),
    };

    let step = if let Some(second) = second {
        second - start
    } else if start > F::ZERO && start < F::ONE {
        if start < end {
            start
        } else {
            -start
        }
    } else {
        if start < end {
            F::ONE
        } else {
            -F::ONE
        }
    };
    if start > end && step > F::ZERO {
        return Err(EvalexprError::CustomMessage(format!(
            "Invalid range params: When step {step} is positive, start: {start} must be less than \
             end: {end}"
        )));
    } else if start < end && step < F::ZERO {
        return Err(EvalexprError::CustomMessage(format!(
            "Invalid range params: When step {step} is negative, start: {start} must be greater \
             than end: {end}"
        )));
    };

    let estimated_len = ((end - start) / step).to_f64() as usize;
    let mut result = thin_vec::ThinVec::with_capacity(estimated_len.clamp(1, 1024));
    result.push(Value::Float(start));
    let mut next = start + step;

    if step > F::ZERO {
        while next <= end {
            result.push(Value::Float(next));
            next = next + step;
        }
    } else {
        while next >= end {
            result.push(Value::Float(next));
            next = next + step;
        }
    }
    Ok(result)
}
#[inline(always)]
pub fn eval_range_with_step<F: EvalexprFloat>(
    start: Value<F>,
    end: Value<F>,
    step: Value<F>,
) -> EvalexprResult<ThinVec<Value<F>>, F> {
    let step = step.as_float()?;
    let end = end.as_float()?;
    let start = start.as_float()?;
    if start > end && step > F::ZERO {
        return Err(EvalexprError::CustomMessage(format!(
            "Invalid range params: When step {step} is positive, start: {start} must be less than \
             end: {end}"
        )));
    } else if start < end && step < F::ZERO {
        return Err(EvalexprError::CustomMessage(format!(
            "Invalid range params: When step {step} is negative, start: {start} must be greater \
             than end: {end}"
        )));
    };

    let estimated_len = ((end - start) / step).to_f64() as usize;
    let mut result = thin_vec::ThinVec::with_capacity(estimated_len.clamp(1, 1024));
    result.push(Value::Float(start));
    let mut next = start + step;
    if step > F::ZERO {
        while next <= end {
            result.push(Value::Float(next));
            next = next + step;
        }
    } else {
        while next >= end {
            result.push(Value::Float(next));
            next = next + step;
        }
    }
    Ok(result)
}

