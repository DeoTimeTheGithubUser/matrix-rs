#![feature(generic_const_exprs)]

use checks::{usize::Zero, Failed};

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Matrix<const R: usize, const C: usize>([[f32; C]; R])
where
    Zero<R>: Failed,
    Zero<C>: Failed;

pub type SquareMatrix<const D: usize> = Matrix<D, D>;
pub type VecMatrix = Vec<Vec<f32>>;

#[macro_export]
macro_rules! matrix {
    () => (compile_error!("Empty matrix not allowed"));
    ($($($value:expr)*),*) => {
        Matrix::from([
            $([$($value),*],)*
        ])
    };
}

impl<const R: usize, const C: usize> Matrix<R, C>
where
    Zero<R>: Failed,
    Zero<C>: Failed,
{
    pub fn new(closure: impl Fn(usize, usize) -> f32) -> Self {
        Self(std::array::from_fn(|row| {
            std::array::from_fn(|column| closure(row, column))
        }))
    }

    pub fn zero() -> Self {
        Self::new(|_, _| 0.0)
    }
    pub const fn is_square(&self) -> bool {
        R == C
    }

    pub fn rows(&self) -> [[f32; C]; R] {
        self.0
    }
    pub fn columns(&self) -> [[f32; R]; C] {
        std::array::from_fn(|i| self.rows().map(|row| row[i]))
    }

    pub fn transpose(&self) -> Matrix<C, R> {
        Matrix::from(self.columns())
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self::new(|r, c| f(self[r][c]))
    }

    pub fn merge<F>(&self, other: Matrix<R, C>, f: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        Self::new(|r, c| f(self[r][c], other[r][c]))
    }
}

impl<const D: usize> SquareMatrix<D>
where
    Zero<D>: Failed,
{
    pub fn identity() -> Self {
        Self::new(|row, column| if row == column { 1.0 } else { 0.0 })
    }

    pub fn determinant(&self) -> f32 {
        Self::determinant_vec_impl(&self.into())
    }

    pub fn has_inverse(&self) -> bool {
        self.determinant() != 0.0
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 {
            None
        } else {
            todo!()
        }
    }

    fn determinant_vec_impl(vec: &VecMatrix) -> f32 {
        let side_len = vec.len();
        match side_len {
            0 => 1.0,
            1 => vec[0][0],
            2 => (vec[0][0] * vec[1][1]) - (vec[0][1] * vec[1][0]),
            _ => {
                let mut det = 0.0;
                let main_row = &vec[0];
                for i in 0..vec.len() {
                    let to = side_len - 1;
                    let sub: VecMatrix = (0..to)
                        .map(|ri| {
                            (0..to)
                                .map(|ci| {
                                    let row = &vec[ri + 1];
                                    row[if ci >= i { ci + 1 } else { ci }]
                                })
                                .collect()
                        })
                        .collect();
                    det += (main_row[i] * Self::determinant_vec_impl(&sub))
                        * (if i % 2 == 0 { 1.0 } else { -1.0 })
                }
                det
            }
        }
    }
}

macro_rules! matrix_merge_op {
    ($type:path => $op:tt) => {
        impl<const R: usize, const C: usize> $type for Matrix<R, C>
        where
            Zero<R>: Failed,
            Zero<C>: Failed,
        {
            type Output = Self;

            fn $op(self, rhs: Self) -> Self::Output {
                self.merge(rhs, |a, b| a.$op(b))
            }
        }
    };
}

matrix_merge_op!(std::ops::Add => add);
matrix_merge_op!(std::ops::Sub => sub);

impl<const R: usize, const C: usize, const C2: usize> std::ops::Mul<Matrix<C, C2>> for Matrix<R, C>
where
    Zero<R>: Failed,
    Zero<C>: Failed,
    Zero<C2>: Failed,
{
    type Output = Matrix<R, C2>;

    fn mul(self, other: Matrix<C, C2>) -> Self::Output {
        Matrix::new(|ri, ci| {
            let row = self.rows()[ri];
            let column = other.columns()[ci];
            let mut sum = 0.0;
            for i in 0..C {
                sum += row[i] * column[i];
            }
            sum
        })
    }
}

impl<const R: usize, const C: usize> std::ops::Mul<f32> for Matrix<R, C>
where
    Zero<R>: Failed,
    Zero<C>: Failed,
{
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        self.map(|v| v * rhs)
    }
}

macro_rules! matrix_from_2d_num_array {
    ($($num:ty)*) => ($(
        impl<const R: usize, const C: usize> From<[[$num; C]; R]> for Matrix<R, C>
        where
            Zero<R>: Failed,
            Zero<C>: Failed
        {
            fn from(value: [[$num; C]; R]) -> Self {
                Self(value.map(|a| a.map(|b| b as f32)))
            }
        }
    )*)
}

matrix_from_2d_num_array!(f32 i32 usize);

impl<const R: usize, const C: usize> From<&Matrix<R, C>> for VecMatrix
where
    Zero<R>: Failed,
    Zero<C>: Failed,
{
    fn from(val: &Matrix<R, C>) -> Self {
        val.rows().map(|r| r.to_vec()).to_vec()
    }
}

impl<const R: usize, const C: usize> Default for Matrix<R, C>
where
    Zero<R>: Failed,
    Zero<C>: Failed,
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<const R: usize, const C: usize> std::ops::Index<usize> for Matrix<R, C>
where
    Zero<R>: Failed,
    Zero<C>: Failed,
{
    type Output = [f32; C];

    fn index(&self, row: usize) -> &Self::Output {
        &self.0[row]
    }
}

impl<const R: usize, const C: usize> std::fmt::Display for Matrix<R, C>
where
    Zero<R>: Failed,
    Zero<C>: Failed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lines = self.rows().map(|row| format!("{:?}", row));
        let longest = lines.iter().map(|s| s.len()).max().unwrap_or(0);
        writeln!(
            f,
            "{:^len$}",
            format!("({}x{} matrix)", R, C),
            len = longest
        )?;
        for line in lines {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}
