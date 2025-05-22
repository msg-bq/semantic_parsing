#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： juzipi
# datetime： 2021/12/30
# software： PyCharm
# description：
# contact： 1129501586@qq.com
# blog: https://piqiandong.blog.csdn.net

import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for ChineseTextEDA.")

with open("readme.md", encoding='utf8') as f:
    readme = f.read()


def do_setup(p_data):
    setup(name='ChineseTextEDA',  # 项目名称（包名）
          version='0.1.0',  # 版本号
          description='en: ChineseTextEDA',  # 包的概括描述
          classifiers=[
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Topic :: Scientific/Engineering :: Artificial Intelligence",
          ],
          install_requires=[
              'synonyms',
              'jieba',
              "tqdm",
          ],
          scripts=['example.py'],
          dependency_links=[],
          long_description=readme,
          long_description_content_type="text/markdown",
          url='',
          author='juzipi',
          author_email='1129501586@qq.com',
          python_requires='>=3.5, <4',
          packages=find_packages(),  # 包列表名
          package_data=p_data,
          zip_safe=False,
          include_package_data=True
          )


def get_files(path, relative_to='toolkit'):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.extend(os.path.join(root, file))
    return all_files


if __name__ == '__main__':
    package_data = {
        "toolkit": (
            get_files("toolkit")
        )
    }
    do_setup(package_data)