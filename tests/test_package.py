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


import unittest

from aequilibrium import __version__


class TestCompute(unittest.TestCase):

    # This function runs once before all the tests in this file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # This function runs once prior to each test
    def setUp(self, *args, **kwargs) -> None:
        return super().setUp(*args, **kwargs)

    def test_valid_version(self):
        self.assertEqual(__version__, "1.0.1")


if __name__ == "__main__":
    unittest.main()
