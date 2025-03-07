# Project State

## Recent Changes

### Fixed Git Repository

**Issue:** The git repository had issues that needed to be fixed with a clean initialization and force push.

**Solution:**

1. Initialized a new git repository in the current directory
2. Renamed the default branch to 'main'
3. Added all files and created an initial commit
4. Set up the remote to point to GitHub
5. Force pushed the repository to GitHub

**Result:** The repository now has a clean git history and is properly connected to GitHub.

### Added Explicit Directory Instructions

**Issue:** The project instructions did not clearly specify that users should work in a dedicated directory, which led to issues with large files being added to the repository when run in system folders.

**Solution:**

1. Added explicit warnings and instructions in multiple files:
   - Added warning boxes and clear instructions in `README.md`
   - Added a critical instruction section in `public/docs/installation.md`
   - Added dedicated directory structure to the scope in `public/docs/prd.md`
   - Added a warning box with CSS styling in `public/index.html`

2. Changed Miniconda installation instructions to use a one-step method that doesn't save the installer file:

   ```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash
   ```

**Result:** The instructions now clearly emphasize the importance of working in a dedicated directory (`~/ai-training`) and not running the project directly in system folders like Documents or Downloads.

### Created Automated Scripts for Large File Removal

**Issue:** Users need an easy way to remove the large Miniconda installer file from Git history to fix the GitHub file size limit error.

**Solution:**

1. Created two automated scripts to handle the large file removal process:
   - `fix-large-file.sh` - Bash script for macOS/Linux users
   - `fix-large-file.bat` - Batch file for Windows users

2. The scripts include:
   - Automatic detection and installation of git-filter-repo if needed
   - Creation of a backup branch for safety
   - Removal of the large file from Git history
   - Verification that the file is gone
   - Clear instructions for pushing changes
   - Guidance on preventing future issues

**Result:** Users now have a simple, one-click solution to fix the GitHub large file issue without having to manually run complex Git commands.

### Fixed GitHub Large File Issue

**Issue:** The repository included a large file (Miniconda3-latest-MacOSX-arm64.sh, 111.77 MB) that exceeds GitHub's file size limit of 100MB, causing push operations to fail.

**Solution:**

1. Updated `.gitignore` to exclude the Miniconda installer and other installer files:

   ```bash
   # Installers
   Miniconda3-latest-MacOSX-arm64.sh
   *.exe
   *.msi
   *.pkg
   *.dmg
   *.deb
   *.rpm
   ```

2. Modified installation instructions in `public/docs/installation.md` to download and run the installer in one step without saving it to the repository:

   ```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash
   ```

3. Created a guide (`git-fix-large-file.md`) with detailed instructions for removing the large file from Git history using various methods:
   - Using git-filter-repo (recommended)
   - Using BFG Repo-Cleaner
   - Manual approach with orphan branch

**Result:** The repository can now be pushed to GitHub without encountering file size limit errors, and future installations won't add large installer files to the repository.

### Updated Installation and Documentation

**Changes:**

1. Added instructions to create a new `/ai-training` folder at the beginning of the installation process
2. Removed all references to Visual Studio Code and Jupyter integration from:
   - `public/docs/installation.md`
   - `public/docs/prd.md`
   - `public/index.html`
3. Simplified the "Running Jupyter Notebooks" section to focus only on the classic Jupyter interface

**Result:** The documentation now provides a more streamlined installation process with a dedicated project directory and focuses exclusively on the classic Jupyter Notebook interface for a more consistent learning experience.

### Fixed HTML Entity Decoding in Copy Functionality

**Issue:** When using the copy button on code blocks, HTML entities (like `&quot;`) were not being decoded to their original characters (like `"`), resulting in code snippets with HTML entities in the clipboard.

**Solution:**

1. Added a new `_decodeHtml` method to the `MarkdownRenderer` class that converts HTML entities back to their original characters using the browser's built-in decoding capabilities:

   ```javascript
   _decodeHtml(html) {
     const textarea = document.createElement('textarea');
     textarea.innerHTML = html;
     return textarea.value;
   }
   ```

2. Updated the `_copyTextToClipboard` method to decode the HTML entities before copying:

   ```javascript
   // Decode HTML entities before copying
   const decodedText = this._decodeHtml(text);
   
   // Create a temporary textarea element
   const textarea = document.createElement('textarea');
   textarea.value = decodedText;
   ```

**Result:** Now when a user clicks the copy button on a code block containing HTML entities (like `&quot;`), they will be properly decoded to their original characters (like `"`) before being copied to the clipboard. This ensures that code snippets like `python -c "import mlx; print(mlx.__version__)"` are copied correctly with proper quotation marks instead of HTML entities.

## Current Features

- Markdown rendering with support for:
  - Headers
  - Paragraphs
  - Code blocks with syntax highlighting
  - Inline code
  - Bold and italic text
  - Links
  - Images
  - Ordered and unordered lists
  - Tables
  - Horizontal rules
- Copy to clipboard functionality for code blocks
- Line numbers for code blocks
- Hash-based navigation for markdown links
- GitHub repository URL conversion for folder links
