"""
AINFT

This script provides functionality to verify the authenticity of an NFT by comparing its on-chain data with a locally generated version. It retrieves the NFT's seed and image from the blockchain, generates a new image using the same seed, and compares the two images to ensure they match. This process helps validate that the NFT's visual representation is consistent with its on-chain data.
"""

import argparse
from minter import Minter
import numpy as np
from web3 import Web3
import requests
from PIL import Image
import io


def verify_chain(provider_url: str, contract_address: str, token_id: int):
    # Initialize Web3
    w3 = Web3(Web3.HTTPProvider(provider_url))

    # Create contract instance
    contract = w3.eth.contract(
        address=contract_address,
        abi=[
            {
                "inputs": [
                    {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
                ],
                "name": "seedOf",
                "outputs": [{"internalType": "uint64", "name": "", "type": "uint64"}],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
                ],
                "name": "tokenURI",
                "outputs": [{"internalType": "string", "name": "", "type": "string"}],
                "stateMutability": "view",
                "type": "function",
            },
        ],
    )

    # Call seedOf function
    seed = contract.functions.seedOf(token_id).call()

    # Get token URI
    token_uri = contract.functions.tokenURI(token_id).call()

    # download json from uri
    response = requests.get(token_uri)
    json_data = response.json()

    # get image from json
    image_uri = json_data["image"]

    # download image from uri
    response = requests.get(image_uri)
    image_data = response.content

    # convert image_data to PIL image
    contract_image = Image.open(io.BytesIO(image_data))

    # generate image from seed
    minter = Minter()
    minted_image, prompt = minter(seed, return_prompt=True)

    print(f"Prompt: {prompt}")

    minted_np = np.array(minted_image)
    contract_np = np.array(contract_image)

    if np.mean((minted_np - contract_np) ** 2) < 0.02:
        print("✅ Same as chain")
        print(f"Token URI: {token_uri}")
    else:
        print("❌ Different as chain")
        print(f"Token URI: {token_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify an NFT on chain")
    parser.add_argument(
        "--provider_url", type=str, required=True, help="Web3 provider URL"
    )
    parser.add_argument(
        "--contract_address",
        type=str,
        help="NFT contract address",
        default="0xaF63754FFDCCEFd9d18cF3e1dd96FD013572164a",
    )
    parser.add_argument(
        "--token_id", type=int, required=True, help="Token ID of the NFT to verify"
    )

    args = parser.parse_args()
    verify_chain(args.provider_url, args.contract_address, args.token_id)