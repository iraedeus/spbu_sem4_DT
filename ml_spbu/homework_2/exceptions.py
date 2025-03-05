"""Module defining custom exception classes for machine learning models.

This module provides custom exception classes to handle specific error conditions
that may occur during the use of machine learning models.
"""


class NotFittedError(Exception):
    """Exception raised when an estimator is used before calling `fit`.

    This exception is designed to be raised by estimators (like classifiers,
    scalers, etc.) when their methods such as `predict`, `transform`, etc.,
    are called before the estimator has been fitted to training data using the
    `fit` method. It indicates that the estimator is in an invalid state for
    performing the requested operation.

    Attributes:
        message (str):  A descriptive message about the error.
                          Defaults to a standard message indicating the need to call 'fit'.
    """

    def __init__(
        self,
        message="This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
    ):
        """Initialize NotFittedError with an optional custom message.

        Args:
            message (str, optional): Custom error message.
                Defaults to "This instance is not fitted yet.
                 Call 'fit' with appropriate arguments before using this estimator.".
        """

        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """Return a string representation of the instance.

        Returns:
            str: A formatted error message including the class name and the specific message.
        """

        return f"NotFittedError: {self.message}"
