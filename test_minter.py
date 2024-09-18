from minter import Minter
from PIL import Image
import numpy as np


def test_minter():
    """
    Check local setup

    If everything is setup correctly, this should exactly reproduced the image
    with the seed 1337 provided in the repo as 1337.png
    """
    minter = Minter()
    minted_image = minter(1337)

    test_image = Image.open("1337.png")

    minted_np = np.array(minted_image)
    test_np = np.array(test_image)

    assert np.mean((minted_np - test_np) ** 2) < 0.02
    print("✅ Machine is ok to verify")


if __name__ == "__main__":
    test_minter()
