"""Dynamic keyword bundles for parameterized public-API tests."""

from typing import Any, TypeAlias

# Parametrized tests also supply deliberately invalid keyword values.
RegistrationOptions: TypeAlias = dict[str, Any]
