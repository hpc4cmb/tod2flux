# tod2flux

Tools for estimating point source flux densities from Time Ordered Data

# Installation

## Get the data files

If you installed the tools from the git repository and got error messages trying to read the Planck data files, you probably do not have the git-lfs (large file storage) client installed. Please see https://git-lfs.github.com.

In short, you will need to install the `git-lfs` client, and configure your local repository to use it:

```
git-lfs install
git-lfs fetch
git-lfs checkout
```

## Install the software

`pip install . [--prefix <PREFIX>]`