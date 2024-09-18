# AINFT

AINFT is a new type of NFT. Instead of representing a link to an image, an AINFT represents the algorithm to create the image. Anyone can use the seed in the AINFT to recreate the image and verify the NFT, even if the original link no longer exists. More information about AINFT can be found on the [website](http://ainft.website) or the [white paper](http://ainft.website/paper.pdf)

This repo aims to verify the authenticity of an AINFT by comparing its on-chain data with a locally generated version. It retrieves the AINFT's seed and image from the blockchain, generates a new image using the same seed, and compares the two images to ensure they match. This process helps validate that the NFT's visual representation is consistent with its on-chain data.

## Features

* Retrieves the NFT's seed and image from the blockchain
* Generates a new image using the same seed
* Compares the two images to ensure they match
* Validates the NFT's visual representation against its on-chain data

## AINFT verification

1. Install the required libraries using pip: `pip install -r requirements.txt`
2. Test your local setup by running `python test_minter.py`. If this script fails, then your setup cannot be used to verify AINFTs.
3. Run the script with the following arguments:
	* `provider_url`: The URL of the Ethereum provider (e.g., Infura, Alchemy)
	* `contract_address`: The address of the NFT contract
	* `token_id`: The ID of the NFT to verify

Example: `python compare_chain.py --provider_url <provider_url> --contract_address <contract_address> --token_id <token_id>`


