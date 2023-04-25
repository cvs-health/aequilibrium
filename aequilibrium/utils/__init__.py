# Copyright 2022 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import randint
from typing import Optional


def random_state_generator(
    random_state: Optional[int] = None, min: int = 0, max: int = 99999
) -> int:
    """Generate a valid random state

    Args:
        random_state (Optional[int]): The provided random state. Defaults to None.
        min (int): The minimum value of a generated random state. Defaults to 0.
        max (int): The maximum value of a generated random state. Defaults to 0.

    Returns:
        int: A valid integer to be used as a random state for reproducibility
    """
    if random_state is None:
        return randint(min, max)
    else:
        return random_state
