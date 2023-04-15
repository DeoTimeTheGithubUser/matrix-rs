#[derive(Debug, PartialEq)]
pub struct Matrix<const R: usize, const C: usize>([[f32; C]; R]);

pub type SquareMatrix<const D: usize> = Matrix<D, D>;
pub type VecMatrix = Vec<Vec<f32>>;

#[macro_export]
macro_rules! matrix {
    () => (compile_error!("Empty matrix is weird and shouldn't really be created."));
    ($($($value:expr)*),*) => {
        Matrix::from([
            $([$($value),*],)*
        ])
    };
}

impl<const R: usize, const C: usize> Matrix<R, C> {
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

impl<const D: usize> SquareMatrix<D> {
    pub fn identity() -> Self {
        Self::new(|row, column| if row == column { 1.0 } else { 0.0 })
    }

    pub fn determinant(&self) -> f32 {
        determinant_vec_impl(&self.into())
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
                det += (main_row[i] * determinant_vec_impl(&sub))
                    * (if i % 2 == 0 { 1.0 } else { -1.0 })
            }
            det
        }
    }
}

macro_rules! matrix_merge_op {
    ($type:path => $op:tt) => {
        impl<const R: usize, const C: usize> $type for Matrix<R, C> {
            type Output = Self;

            fn $op(self, rhs: Self) -> Self::Output {
                self.merge(rhs, |a, b| a.$op(b))
            }
        }
    };
}

matrix_merge_op!(std::ops::Add => add);
matrix_merge_op!(std::ops::Sub => sub);

impl<const R: usize, const C: usize, const C2: usize> std::ops::Mul<Matrix<C, C2>>
    for Matrix<R, C>
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

impl<const R: usize, const C: usize> std::ops::Mul<f32> for Matrix<R, C> {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        self.map(|v| v * rhs)
    }
}

macro_rules! matrix_from_2d_num_array {
    ($($num:ty)*) => ($(
        impl<
            const R: usize,
            const C: usize
        > From<[[$num; C]; R]> for Matrix<R, C> {
            fn from(value: [[$num; C]; R]) -> Self {
                Self(value.map(|a| a.map(|b| b as f32)))
            }
        }
    )*)
}

matrix_from_2d_num_array!(f32 i32 usize);

impl<const R: usize, const C: usize> Into<VecMatrix> for &Matrix<R, C> {
    fn into(self) -> VecMatrix {
        self.rows().map(|r| r.to_vec()).to_vec()
    }
}

impl<const R: usize, const C: usize> Default for Matrix<R, C> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const R: usize, const C: usize> std::ops::Index<usize> for Matrix<R, C> {
    type Output = [f32; C];

    fn index(&self, row: usize) -> &Self::Output {
        &self.0[row]
    }
}

impl<const R: usize, const C: usize> std::fmt::Display for Matrix<R, C> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_3x3_1to9() -> Matrix<3, 3> {
        matrix! {
            1 2 3,
            4 5 6,
            7 8 9
        }
    }

    #[test]
    fn test_matrix_macro() {
        let macro_matrix = matrix! {
            99   12     0,
            55   33485  30
        };
        assert_eq!(macro_matrix, Matrix::from([[99, 12, 0], [55, 33485, 30]]))
    }

    #[test]
    fn test_matrix_display() {
        println!("{}", matrix_3x3_1to9())
    }

    #[test]
    fn test_matrix_zero() {
        let zero_matrix = SquareMatrix::<3>::zero();
        assert_eq!(
            zero_matrix,
            matrix! {
                0 0 0,
                0 0 0,
                0 0 0
            }
        );
    }

    #[test]
    fn test_matrix_identity() {
        let identity_matrix = SquareMatrix::<3>::identity();
        assert_eq!(
            identity_matrix,
            matrix! {
                1 0 0,
                0 1 0,
                0 0 1
            }
        );
    }

    #[test]
    fn test_matrix_is_square() {
        let square_matrix = SquareMatrix::<5>::zero();
        assert!(square_matrix.is_square(), "Square matrix not square (how).")
    }

    #[test]
    fn test_matrix_addition() {
        let m1 = matrix_3x3_1to9();
        let m2 = matrix_3x3_1to9();
        let m3 = m1 + m2;

        assert_eq!(
            m3,
            matrix! {
                2  4  6,
                8  10 12,
                14 16 18
            }
        );
    }

    #[test]
    fn test_matrix_subtraction() {
        let m1 = matrix_3x3_1to9();
        let m2 = matrix_3x3_1to9();
        let m3 = m1 - m2;
        assert_eq!(m3, Matrix::zero());
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1_a = matrix! { 5 };
        let m1_b = matrix! { 3 };
        assert_eq!(
            m1_a * m1_b,
            Matrix::from([[15]]),
            "(1x1) * (1x1) matrix multiplication failed."
        );

        let m2_a = matrix! {
            1 2 3,
            4 5 6
        };
        let m2_b = matrix! {
            1 2,
            3 4,
            5 6
        };

        assert_eq!(
            m2_a * m2_b,
            matrix! {
                22 28,
                49 64
            },
            "(2x3) * (3x2) matrix multiplication failed."
        );
    }

    #[test]
    fn test_matrix_determinant() {
        let m0 = Matrix([]);
        assert_eq!(m0.determinant(), 1.0, "(0x0) matrix determinant failed.");

        let m1 = matrix! { 123 };
        assert_eq!(m1.determinant(), 123.0, "(1x1) matrix determinant failed.");


        let m2 = matrix! {
            11 7,
            2  5
        };
        assert_eq!(m2.determinant(), 41.0, "(2x2) matrix determinant failed.");


        let m3 = matrix! {
            7 4  5,
            3 10 1,
            9 0  7
        };
        assert_eq!(m3.determinant(), -8.0, "(3x3) matrix determinant failed.");


        let m4 = matrix! {
            7 8  4 5,
            6 22 1 4,
            7 12 3 2,
            0 5  9 3
        };
        assert_eq!(m4.determinant(), 2921.0, "(4x4) matrix determinant failed.");
    }

    #[test]
    fn test_matrix_inverse() {
        let matrix_where_determinant_is_0 = SquareMatrix::<3>::zero();
        assert!(!matrix_where_determinant_is_0.has_inverse());
    }

    #[test]
    fn test_matrix_transpose() {
        let transposed_matrix = matrix! {
            1 2 3,
            4 5 6
        }.transpose();
        assert_eq!(
            transposed_matrix,
            matrix! {
                1 4,
                2 5,
                3 6
            }
        )
    }
}
