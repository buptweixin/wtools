"""Tests for wtools.utils.utils -- isnotebook and utility classes."""

from unittest.mock import patch

import pytest

from wtools.utils.utils import isnotebook


class TestIsNotebook:
    def test_returns_false_in_standard_python(self):
        """In a standard Python interpreter (no IPython),
        get_ipython raises NameError and isnotebook returns False.
        """
        # get_ipython is a builtin injected by IPython; in a test runner
        # it should not be available, so NameError is caught -> False.
        result = isnotebook()
        assert result is False

    def test_returns_true_for_zmq_shell(self):
        """When IPython is available and the shell class is
        ZMQInteractiveShell (Jupyter notebook), should return True.
        """
        fake_shell = type("ZMQInteractiveShell", (), {})

        class FakeIPython:
            __class__ = fake_shell

        with patch("wtools.utils.utils.get_ipython", return_value=FakeIPython()):
            assert isnotebook() is True

    def test_returns_false_for_terminal_shell(self):
        """When IPython is available but it's a terminal session
        (TerminalInteractiveShell), should return False.
        """
        fake_shell = type("TerminalInteractiveShell", (), {})

        class FakeIPython:
            __class__ = fake_shell

        with patch("wtools.utils.utils.get_ipython", return_value=FakeIPython()):
            assert isnotebook() is False

    def test_returns_false_for_unknown_shell(self):
        """For any other IPython shell class, should return False."""
        fake_shell = type("SomeOtherShell", (), {})

        class FakeIPython:
            __class__ = fake_shell

        with patch("wtools.utils.utils.get_ipython", return_value=FakeIPython()):
            assert isnotebook() is False

    def test_returns_false_when_get_ipython_raises_name_error(self):
        """When get_ipython is not defined (NameError), returns False."""

        def raise_name_error():
            raise NameError("get_ipython")

        with patch("wtools.utils.utils.get_ipython", side_effect=raise_name_error):
            assert isnotebook() is False
