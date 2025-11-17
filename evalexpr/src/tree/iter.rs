use crate::{operator::Operator, value::numeric_types::EvalexprFloat, Node};
use std::slice::{Iter, IterMut};

/// An iterator that traverses an operator tree in pre-order.
pub struct NodeIter<'a, NumericTypes: EvalexprFloat> {
    node: Option<&'a Node<NumericTypes>>,
    stack: Vec<Iter<'a, Node<NumericTypes>>>,
}

impl<'a, NumericTypes: EvalexprFloat> NodeIter<'a, NumericTypes> {
    fn new(node: &'a Node<NumericTypes>) -> Self {
        NodeIter {
            node: Some(node),
            stack: vec![node.children.iter()],
        }
    }
}

impl<'a, NumericTypes: EvalexprFloat> Iterator for NodeIter<'a, NumericTypes> {
    type Item = &'a Node<NumericTypes>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.node.take() {
            return Some(node);
        }
        loop {
            let mut result = None;

            if let Some(last) = self.stack.last_mut() {
                if let Some(next) = last.next() {
                    result = Some(next);
                } else {
                    // Can not fail because we just borrowed last.
                    // We just checked that the iterator is empty, so we can safely discard it.
                    let _ = self.stack.pop().unwrap();
                }
            } else {
                return None;
            }

            if let Some(result) = result {
                self.stack.push(result.children.iter());
                return Some(result);
            }
        }
    }
}

/// An iterator that mutably traverses an operator tree in pre-order.
pub struct OperatorIterMut<'a, NumericTypes: EvalexprFloat> {
    root: Option<&'a mut Operator<NumericTypes>>,
    stack: Vec<IterMut<'a, Node<NumericTypes>>>,
}

impl<'a, NumericTypes: EvalexprFloat> OperatorIterMut<'a, NumericTypes> {
    fn new(node: &'a mut Node<NumericTypes>) -> Self {
        OperatorIterMut {
            root: Some(&mut node.operator),
            stack: vec![node.children.iter_mut()],
        }
    }
}

impl<'a, NumericTypes: EvalexprFloat> Iterator for OperatorIterMut<'a, NumericTypes> {
    type Item = &'a mut Operator<NumericTypes>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(root) = self.root.take() {
            return Some(root);
        }
        loop {
            let mut result = None;

            if let Some(last) = self.stack.last_mut() {
                if let Some(next) = last.next() {
                    result = Some(next);
                } else {
                    // Can not fail because we just borrowed last.
                    // We just checked that the iterator is empty, so we can safely discard it.
                    let _ = self.stack.pop().unwrap();
                }
            } else {
                return None;
            }

            if let Some(result) = result {
                self.stack.push(result.children.iter_mut());
                return Some(&mut result.operator);
            }
        }
    }
}

impl<NumericTypes: EvalexprFloat> Node<NumericTypes> {
    /// Returns an iterator over all nodes in this tree.
    pub fn iter(&self) -> impl Iterator<Item = &Node<NumericTypes>> {
        NodeIter::new(self)
    }

    /// Returns a mutable iterator over all operators in this tree.
    pub fn iter_operators_mut(&mut self) -> impl Iterator<Item = &mut Operator<NumericTypes>> {
        OperatorIterMut::new(self)
    }
}
