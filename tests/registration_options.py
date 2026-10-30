"""Dynamic keyword bundles for parameterized public-API tests."""

from typing import Any

# Parametrized tests also supply deliberately invalid keyword values.
type RegistrationOptions = dict[str, Any]
