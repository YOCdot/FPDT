# ---*--- File Information ---*---
# @File       :  __init__.py.py
# @Date&Time  :  2022-08-26, 17:30:23
# @Project    :  tfpd
# @Platform   :  Apple ARM64
# @Software   :  PyCharm
# @Author     :  yoc
# @Email      :  yyyyyoc@hotmail.com


from .detector import build


def build_model(args):
    return build(args)
