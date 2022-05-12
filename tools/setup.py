'''
Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
###############################################################################
# setup.py
# You can optionally install this project to your local site-packages directory
###############################################################################
import setuptools

setuptools.setup(
    name='robrank',
    description='Adversarial Robustness in Deep Ranking',
    version='0.1',
    url='https://cdluminate.github.io/robrank/',
    maintainer='Mo Zhou',
    maintainer_email='cdluminate@gmail.com',
    license='Apache-2.0',
    packages=setuptools.find_packages(include=['robrank', 'robrank.*']),
    install_requires=list(x.strip() for x in
                          open('requirements.txt', 'rt').readlines()),
    python_requires='>=3.7',
    zip_safe=True,
)
