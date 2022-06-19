#include <stdio.h>
#include <immintrin.h>

int hammingDistances_SSE(size_t n, const char a[n], const char b[n]) {
    const int iters = n / 16;
    unsigned int res = 0;
    if (n >= 16) {
        for (int i = 0; i < iters; i++) {
            __m128i differences = _mm_setzero_si128();
            const __m128i smm1 = _mm_loadu_si128((__m128i *)&a[i * 16]);
            const __m128i smm2 = _mm_loadu_si128((__m128i *)&b[i * 16]);
            differences = _mm_sub_epi8(differences, _mm_cmpeq_epi8(smm1, smm2));
            differences = _mm_sad_epu8(differences, _mm_setzero_si128());
            int x = (_mm_extract_epi16(differences, 0) + _mm_extract_epi16(differences, 4));
            res += 16 - x;
        }
    }
    for (size_t i = 16 * iters; i < n; i++) {
        res += a[i] != b[i];
    }
    return res;
}

size_t hamming_dist(size_t n, const char a[n], const char b[n]) {
    size_t res = 0;
    for (size_t i = 0; i < n; i++)
        res += a[i] != b[i];
    return res;
}

