/*
 * Copyright 2020, 2021, Ludovic Henry
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Please contact git@ludovic.dev or visit ludovic.dev if you need additional
 * information or have any questions.
 */

package dev.ludovic.netlib.blas;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DgemmOffsetTest {

    @Test
    void testJavaBLASOffsetBoundsCheck() {
        // Test case from GitHub issue #23 - DGEMM with offset incorrectly reports index out of bounds
        // Matrix A: 2 rows, 3 cols, stored starting at offset 1 in a length-9 array
        double[] a = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
        int m = 2; // rows of op(A) and C
        int k = 3; // cols of op(A) and rows of op(B)
        int n = 1; // cols of op(B) and C
        int lda = 3; // leading dimension of A
        int offsetA = 1;

        // Matrix B: 3 rows, 1 col
        double[] b = {1.0, 2.0, 3.0};
        int ldb = 3;
        int offsetB = 0;

        // Output matrix C: 2 rows, 1 col
        double[] c = new double[2];
        int ldc = 2;
        int offsetC = 0;

        // Test with JavaBLAS (which uses AbstractBLAS bounds checking)
        BLAS javaBlas = JavaBLAS.getInstance();
        
        // This should NOT throw IndexOutOfBoundsException
        // The maximum index accessed in matrix A should be: offsetA + (k-1)*lda + (m-1) = 1 + 2*3 + 1 = 8
        // which is valid for an array of length 9 (indices 0-8)
        assertDoesNotThrow(() -> {
            javaBlas.dgemm("N", "N", m, n, k, 1.0, a, offsetA, lda, b, offsetB, ldb, 0.0, c, offsetC, ldc);
        });

        // Also test with F2J implementation to verify the computation is correct
        BLAS f2j = F2jBLAS.getInstance();
        double[] expected = new double[2];
        f2j.dgemm("N", "N", m, n, k, 1.0, a, offsetA, lda, b, offsetB, ldb, 0.0, expected, offsetC, ldc);
        
        // The computation should match
        assertArrayEquals(expected, c, 1e-11);
    }
    
    @Test
    void testF2JBLASOffsetBoundsCheck() {
        // Also test that F2J BLAS works with same parameters (as reference)
        double[] a = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
        int m = 2, k = 3, n = 1, lda = 3, offsetA = 1;
        double[] b = {1.0, 2.0, 3.0};
        int ldb = 3, offsetB = 0;
        double[] c = new double[2];
        int ldc = 2, offsetC = 0;

        BLAS f2j = F2jBLAS.getInstance();
        
        // F2J should work without issues
        assertDoesNotThrow(() -> {
            f2j.dgemm("N", "N", m, n, k, 1.0, a, offsetA, lda, b, offsetB, ldb, 0.0, c, offsetC, ldc);
        });
        
        // Expected result: matrix multiplication of 2x3 * 3x1 = 2x1
        // A (with offset 1) represents matrix [[4,5,6], [7,8,9]]
        // B represents matrix [[1], [2], [3]]
        // Result should be [[4*1+5*2+6*3], [7*1+8*2+9*3]] = [[32], [50]]
        assertEquals(32.0, c[0], 1e-11);
        assertEquals(50.0, c[1], 1e-11);
    }
    
    @Test
    void testEdgeCases() {
        BLAS javaBlas = JavaBLAS.getInstance();
        
        // Test case 1: minimal matrices 1x1 * 1x1 with offset
        double[] a1 = {0.0, 5.0}; // Value at offset 1
        double[] b1 = {0.0, 3.0}; // Value at offset 1  
        double[] c1 = new double[1];
        assertDoesNotThrow(() -> {
            javaBlas.dgemm("N", "N", 1, 1, 1, 1.0, a1, 1, 1, b1, 1, 1, 0.0, c1, 0, 1);
        });
        assertEquals(15.0, c1[0], 1e-11);
        
        // Test case 2: transposed matrix bounds checking  
        double[] aT = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // 3x2 stored in column-major, offset 1
        double[] b2 = {1.0, 2.0};
        double[] c2 = new double[3];
        // A^T is 2x3, B is 2x1, result is 2x1 -> but we want C to be 3x1, so m=3, n=1, k=2
        assertDoesNotThrow(() -> {
            javaBlas.dgemm("T", "N", 3, 1, 2, 1.0, aT, 1, 2, b2, 0, 2, 0.0, c2, 0, 3);
        });
        
        // Test case 3: Both matrices transposed
        double[] bT = {1.0, 3.0, 2.0, 4.0}; // 1x2 stored as 2x1, offset 1
        double[] c3 = new double[2];
        assertDoesNotThrow(() -> {
            javaBlas.dgemm("N", "T", 2, 1, 2, 1.0, aT, 1, 2, bT, 1, 1, 0.0, c3, 0, 2);
        });
    }
}