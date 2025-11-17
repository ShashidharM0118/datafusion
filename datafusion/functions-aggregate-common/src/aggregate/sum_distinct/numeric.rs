// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines the accumulator for `SUM DISTINCT` for primitive numeric types

use std::fmt::Debug;
use std::mem::size_of_val;

use arrow::array::ArrayRef;
use arrow::array::ArrowNativeTypeOp;
use arrow::array::ArrowPrimitiveType;
use arrow::datatypes::ArrowNativeType;
use arrow::datatypes::DataType;

use datafusion_common::Result;
use datafusion_common::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;

use crate::utils::{GenericDistinctBuffer, Hashable};

/// Accumulator for computing SUM(DISTINCT expr)
///
/// Uses Hashable storage to correctly handle floats and other numeric types
#[derive(Debug)]
pub struct DistinctSumAccumulator<T: ArrowPrimitiveType> {
    values: GenericDistinctBuffer<T, datafusion_common::HashSet<Hashable<T::Native>, ahash::RandomState>>,
    data_type: DataType,
}

impl<T: ArrowPrimitiveType> DistinctSumAccumulator<T> {
    pub fn new(data_type: &DataType) -> Self {
        Self {
            values: GenericDistinctBuffer::new(data_type.clone()),
            data_type: data_type.clone(),
        }
    }

    /// Get the number of distinct values stored in the accumulator.
    ///
    /// The returned value is the count of unique values currently held in the internal distinct buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// // `acc` is a DistinctSumAccumulator created elsewhere.
    /// // let acc = DistinctSumAccumulator::new(DataType::Int32);
    /// // assert_eq!(acc.distinct_count(), 0);
    /// ```
    pub fn distinct_count(&self) -> usize {
        self.values.values.len()
    }
}

impl<T: ArrowPrimitiveType + Send + Sync + Debug> Accumulator for DistinctSumAccumulator<T> {
    /// Produces the accumulator's current state as a vector of scalar values.
    ///
    /// The vector contains the serialized representation of the internal distinct buffer and is suitable for persisting or merging accumulator state.
    ///
    /// # Returns
    ///
    /// A `Vec<ScalarValue>` representing the accumulator's distinct-state; empty if no distinct values are stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use datafusion_common::ScalarValue;
    /// use arrow::datatypes::DataType;
    /// // Construct a DistinctSumAccumulator for i64 values and obtain its state.
    /// let mut acc = DistinctSumAccumulator::<arrow::datatypes::Int64Type>::new(DataType::Int64);
    /// let state: Vec<ScalarValue> = acc.state().unwrap();
    /// assert!(state.is_empty());
    /// ```
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.values.state()
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.values.update_batch(values)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.values.merge_batch(states)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.distinct_count() == 0 {
            ScalarValue::new_primitive::<T>(None, &self.data_type)
        } else {
            let mut acc = T::Native::usize_as(0);
            for distinct_value in self.values.values.iter() {
                acc = acc.add_wrapping(distinct_value.0)
            }
            ScalarValue::new_primitive::<T>(Some(acc), &self.data_type)
        }
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.values.size()
    }
}