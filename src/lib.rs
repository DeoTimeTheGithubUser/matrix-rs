#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Matrix<const R: usize, const C: usize>([[i32; C]; R]);

pub type SquareMatrix<const D: usize> = Matrix<D, D>;
pub type VecMatrix = Vec<Vec<i32>>;

impl<
    const R: usize,
    const C: usize
> Matrix<R, C> {
    pub fn new(closure: impl Fn(usize, usize) -> i32) -> Self {
        Self(
            std::array::from_fn(|row|
                std::array::from_fn(|column| closure(row, column))
            )
        )
    }

    pub fn zero() -> Self { Self::new(|_, _| 0) }
    pub const fn is_square(self) -> bool { R == C }

    pub fn rows(self) -> [[i32; C]; R] { self.0 }
    pub fn columns(self) -> [[i32; R]; C] {
        std::array::from_fn(|i| self.rows().map(|row| row[i]))
    }

    pub fn transform<F>(self, f: F) -> Self
        where F: Fn(i32) -> i32 {
        Self::new(|r, c| f(self[r][c]))
    }

    pub fn merge<F>(self, other: Matrix<R, C>, f: F) -> Self
        where F: Fn(i32, i32) -> i32 {
        Self::new(|r, c| f(self[r][c], other[r][c]))
    }
}

impl<const D: usize> SquareMatrix<D> {
    pub fn identity() -> Self {
        Self::new(|row, column| if row == column { 1 } else { 0 })
    }

    pub fn inverse(self) -> Option<Self> {
        todo!()
    }

    pub fn determinant(self) -> i32 {
        determinant_vec_impl(self.into())
    }
}

fn determinant_vec_impl(vec: VecMatrix) -> i32 {
    let side_len = vec.len();
    match side_len {
        0 => 1,
        1 => vec[0][0],
        2 => (vec[0][0] * vec[1][1]) - (vec[0][1] * vec[1][0]),
        _ => {
            let mut det = 0;
            let main_row = &vec[0];
            for i in 0..vec.len() {
                let to = side_len - 1;
                let sub: VecMatrix = (0..to).map(|ri|
                    (0..to).map(|ci| {
                        let row = &vec[ri + 1];
                        row[if ci >= i { ci + 1 } else { ci }]
                    }).collect()
                ).collect();
                det += (main_row[i] * determinant_vec_impl(sub))
                    * (if i % 2 == 0 { 1 } else { -1 })
            }
            det
        }
    }
}

#[macro_export]
macro_rules! matrix_merge_op {
    ($type:path, $op:tt) => {
        impl<
            const R: usize,
            const C: usize
        > $type for Matrix<R, C> {
            type Output = Self;

            fn $op(self, rhs: Self) -> Self::Output {
                self.merge(rhs, |a, b| a.$op(b))
            }
        }
    }
}

matrix_merge_op!(std::ops::Add, add);
matrix_merge_op!(std::ops::Sub, sub);

impl<
    const R: usize,
    const C: usize,
    const C2: usize
> std::ops::Mul<Matrix<C, C2>> for Matrix<R, C> {
    type Output = Matrix<R, C2>;

    fn mul(self, other: Matrix<C, C2>) -> Self::Output {
        Matrix::new(|ri, ci| {
            let row = self.rows()[ri];
            let column = other.columns()[ci];
            let mut sum = 0;
            for i in 0..C {
                sum += row[i] * column[i];
            }
            sum
        })
    }
}

impl<
    const R: usize,
    const C: usize,
> std::ops::Mul<i32> for Matrix<R, C> {
    type Output = Self;
    fn mul(self, rhs: i32) -> Self::Output {
        self.transform(|v| v * rhs)
    }
}

impl<
    const R: usize,
    const C: usize
> From<[[i32; C]; R]> for Matrix<R, C> {
    fn from(value: [[i32; C]; R]) -> Self {
        Self(value)
    }
}

impl<
    const R: usize,
    const C: usize
> Into<VecMatrix> for Matrix<R, C> {
    fn into(self) -> VecMatrix {
        self.rows().map(|r| r.to_vec()).to_vec()
    }
}

impl<
    const R: usize,
    const C: usize
> Default for Matrix<R, C> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<
    const R: usize,
    const C: usize
> std::ops::Index<usize> for Matrix<R, C> {
    type Output = [i32; C];

    fn index(&self, row: usize) -> &Self::Output {
        &self.0[row]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn standard_matrix() -> Matrix<3, 3> {
        Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
    }

    #[test]
    fn test_matrix_columns() {
        let matrix = standard_matrix();
        assert_eq!(matrix.columns(), [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ])
    }

    #[test]
    fn test_matrix_zero() {
        let matrix = SquareMatrix::<3>::zero();
        assert_eq!(
            matrix,
            Matrix([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        );
    }

    #[test]
    fn test_matrix_identity() {
        let matrix = SquareMatrix::<3>::identity();
        assert_eq!(
            matrix,
            Matrix([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        );
    }

    #[test]
    fn test_matrix_is_square() {
        let matrix = SquareMatrix::<5>::zero();
        assert!(matrix.is_square(), "Square matrix not square.")
    }

    #[test]
    fn test_matrix_addition() {
        let m1 = standard_matrix();
        let m2 = standard_matrix();
        let m3 = m1 + m2;
        assert_eq!(
            m3,
            Matrix([
                [2, 4, 6],
                [8, 10, 12],
                [14, 16, 18]
            ])
        );
    }

    #[test]
    fn test_matrix_subtraction() {
        let m1 = standard_matrix();
        let m2 = standard_matrix();
        let m3 = m1 - m2;
        assert_eq!(m3, Matrix::zero());
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1_a = Matrix::from([[5]]);
        let m1_b = Matrix::from([[3]]);
        assert_eq!(
            m1_a * m1_b,
            Matrix::from([[15]]),
            "(1x1) * (1x1) matrix multiplication failed."
        );

        let m2_a = Matrix::from([
            [1, 2, 3],
            [4, 5, 6]
        ]);
        let m2_b = Matrix::from([
            [1, 2],
            [3, 4],
            [5, 6]
        ]);
        assert_eq!(
            m2_a * m2_b,
            Matrix::from([
                [22, 28],
                [49, 64]
            ]),
            "(2x3) * (3x2) matrix multiplication failed."
        );
    }

    #[test]
    fn test_matrix_determinant() {
        let m0 = SquareMatrix::from([]);
        assert_eq!(m0.determinant(), 1, "(0x0) matrix determinant failed.");

        let m1 = SquareMatrix::from([[123]]);
        assert_eq!(m1.determinant(), 123, "(1x1) matrix determinant failed.");

        let m2 = SquareMatrix::from([
            [11, 7],
            [2, 5]
        ]);
        assert_eq!(m2.determinant(), 41, "(2x2) matrix determinant failed.");

        let m3 = SquareMatrix::from([
            [7, 4, 5],
            [3, 10, 1],
            [9, 0, 7]
        ]);
        assert_eq!(m3.determinant(), -8, "(3x3) matrix determinant failed.");

        let m4 = SquareMatrix::from([
            [7, 8, 4, 5],
            [6, 22, 1, 4],
            [7, 12, 3, 2],
            [0, 5, 14, 3]
        ]);
        assert_eq!(m4.determinant(), 4471, "(4x4) matrix determinant failed.");
    }
}

