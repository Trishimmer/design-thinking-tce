import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus):
    if exitstatus == 0:
        terminalreporter.write("\n🎉 All tests passed successfully! 🎉\n", purple=True)