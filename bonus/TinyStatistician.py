import numpy as np
from math import ceil, floor, sqrt


class TinyStatistician:
    """https://www.mathportal.org/calculators/statistics-calculator/descriptive-statistics-calculator.php"""
    @staticmethod
    def check_input(x):
        """Check if inputs are valid (not empty list or nd.array, and ints or floats elements) """
        if isinstance(x, list):
            if not x:
                return 1
        if isinstance(x, np.ndarray):
            if not x.any():
                return 1
        try:
            assert isinstance(x, list) or isinstance(
                x, np.ndarray), "argument should be a list or array"
            # for i in range(len(x)):
            # 	assert isinstance(x[i], int) or isinstance(x[i], np.int64) or isinstance(x[i], float) or isinstance(
            # x[i], np.float64), "list or arrays should only contains either ints or floats"

        except AssertionError as msg:
            print(msg)
            return(1)
        return 0

    @staticmethod
    def maxx(x):
        if TinyStatistician.check_input(x):
            return None
        m = -1
        for v in x:
            if v > m:
                m = v
        return float(m)

    @staticmethod
    def countx(x):
        if TinyStatistician.check_input(x):
            return None
        m = 0
        for v in x:
            m += 1
        return float(m)

    @staticmethod
    def minx(x):
        if TinyStatistician.check_input(x):
            return None
        m = TinyStatistician.maxx(x)
        for v in x:
            if v < m:
                m = v
        return float(m)

    @staticmethod
    def sumx(x):
        if TinyStatistician.check_input(x):
            return None
        m = 0
        for v in x:
            m += v
        return float(m)

    @staticmethod
    def meanx(x):
        """computes the mean of a given non-empty list or array x"""
        if TinyStatistician.check_input(x):
            return None
        return float(TinyStatistician.sumx(x) / len(x))

    @staticmethod
    def median(x):
        """computes the median of a given non-empty list or array x.
        Median : value separating the higher half from the lower half of a data sample, a population, or a probability distribution. For a data set, it may be thought of as "the middle" value. """
        if TinyStatistician.check_input(x):
            return None
        n = len(x)
        x.sort()
        if (n % 2 == 1):
            return float(x[int(n / 2)])
        else:
            return float((x[int(n / 2) - 1] + x[int(n / 2)]) / 2)

    @staticmethod
    def quartile(x):
        """computes the 1st and 3rd quartiles of a given non-empty array x.
        1st quartile : middle number between the smallest value (minimum) and the median of the data set.
        3rd quartile : middle number between the median and the highest value (maximum) of the data set.
        """
        if TinyStatistician.check_input(x):
            return None
        n = len(x)
        x.sort()

        # # method 1
        # if (n % 2 == 1):
        #     i1 = round(n / 2 * 0.25)
        #     i2 = round((n / 2 + 1) * 0.75)

        if (n % 2 == 1):
            x1 = x[: ceil(n / 2)]
            x3 = x[floor(n / 2):]
            return [TinyStatistician.median(x1), TinyStatistician.median(x3)]
        else:
            x1 = x[: ceil(n / 2)]
            x3 = x[floor(n / 2):]
            return [TinyStatistician.median(x1), TinyStatistician.median(x3)]

        # method 2
        if (n % 2 == 1):
            x1 = x[: floor(n / 2)]
            x3 = x[floor(n / 2) + 1:]
            return [TinyStatistician.median(x1), TinyStatistician.median(x3)]
        else:
            x1 = x[: floor(n / 2)]
            x3 = x[floor(n / 2):]
            return [TinyStatistician.median(x1), TinyStatistician.median(x3)]

    @staticmethod
    def percentile(x, p):
        """computes the expected percentile of a given non-empty list or array x.
        The method returns the percentile as a float, otherwise None if x is an empty list or array or a non expected type object.
        The second parameter is the wished percentile.
        The kth percentile of a set of data is the value at which k percent of the data is below it.
        https://www.calculatorsoup.com/calculators/statistics/percentile-calculator.php
        https://www.translatorscafe.com/unit-converter/fr-FR/calculator/percentile/
        """
        if TinyStatistician.check_input(x):
            return None
        if (not isinstance(p, int) or p < 0 or p > 100):
            return None
        n = len(x)
        x.sort()

        #  rank r for the percentile p we want to find:
        r = (p / 100) * (n - 1)
        #  p is then interpolated using ri, the integer part of r, and rf, the fractional part of r:
        ri = floor(r)
        rf = r - ri
        percentile = x[ri] + rf * (x[ri + 1] - x[ri])
        # rc = ceil(r)
        # percentile = x[ri] * (ri - r) + x[rc] * (r - rc)
        return float(percentile)

    @staticmethod
    def var(x):
        """computes the variance of a given non-empty list or array x.
        Variance: expectation of the squared deviation of a random variable from its population mean or sample mean. Variance is a measure of dispersion, meaning it is a measure of how far a set of numbers is spread out from their average value. """
        if TinyStatistician.check_input(x):
            return None
        n = len(x)
        mu = TinyStatistician.meanx(x)
        # method 1 (correction-de-bessel)
        # In statistics, Bessel's correction is the use of n − 1 instead of n in the formula for the sample variance and sample standard deviation,[1] where n is the number of observations in a sample. This method corrects the bias in the estimation of the population variance. It also partially corrects the bias in the estimation of the population standard deviation.
        return float(sum([(v - mu)**2 for v in x]) / (n - 1))
        # method 2
        return float(sum([(v - mu)**2 for v in x]) / n)

    @staticmethod
    def std(x):
        """computes the standard deviation of a given non-empty list or array x.
        STD: measure of the amount of variation or dispersion of a set of values.
        """
        if TinyStatistician.check_input(x):
            return None
        return float(sqrt(TinyStatistician.var(x)))
