 /* 
 * Markdown Renderer
 * A simple script to render markdown content as HTML
 */

// Simple markdown parser
class MarkdownParser {
  constructor() {
    // Define regex patterns for markdown syntax
    this.patterns = {
      header: /^(#{1,6})\s+(.+)$/gm,
      paragraph: /^(?!<[a-z]|#{1,6}\s|```|\*|\d+\.\s|\-\s|\|)(.+)$/gm,
      codeBlock: /```([a-z]*)\n([\s\S]*?)```/gm,
      inlineCode: /`([^`]+)`/g,
      bold: /\*\*([^*]+)\*\*/g,
      italic: /\*([^*]+)\*/g,
      link: /\[([^\]]+)\]\(([^)]+)\)/g,
      unorderedList: /^\-\s+(.+)$/gm,
      orderedList: /^(\d+)\.\s+(.+)$/gm, // Modified to capture the number
      horizontalRule: /^---$/gm,
      image: /!\[([^\]]+)\]\(([^)]+)\)/g,
      // Table pattern - matches markdown tables
      tableRow: /^\|(.+)\|$/gm,
      tableHeader: /^\|((?:[ \t]*:?-+:?[ \t]*\|)+)$/gm
    };
  }

  /**
   * Parse markdown text to HTML
   * @param {string} markdown - The markdown text to parse
   * @returns {string} The HTML representation of the markdown
   */
  parse(markdown) {
    if (!markdown) return '';
    
    // Pre-process the markdown to clean it up
    let html = this._preprocessMarkdown(markdown);
    
    // Store code blocks to prevent processing their contents
    const codeBlocks = [];
    html = html.replace(this.patterns.codeBlock, (match, language, code) => {
      const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
      codeBlocks.push({
        language: language.trim(),
        code: this._escapeHtml(code.trim())
      });
      return placeholder;
    });
    
    // Process tables
    html = this._processTables(html);
    
    // Process headers
    html = html.replace(this.patterns.header, (match, hashes, content) => {
      const level = hashes.length;
      return `<h${level}>${content.trim()}</h${level}>`;
    });
    
    // Process paragraphs - only match non-empty lines
    html = html.replace(this.patterns.paragraph, (match, content) => {
      if (content.trim()) {
        return `<p>${content.trim()}</p>`;
      }
      return '';
    });
    
    // Process inline elements
    html = html.replace(this.patterns.inlineCode, (match, code) => {
      return `<code>${this._escapeHtml(code)}</code>`;
    });
    
    html = html.replace(this.patterns.bold, '<strong>$1</strong>');
    html = html.replace(this.patterns.italic, '<em>$1</em>');
    
    // Process links - Check if the link is to a markdown file or a folder
    html = html.replace(this.patterns.link, (match, text, url) => {
      // Check if the URL ends with .md
      if (url.endsWith('.md')) {
        // Extract the document name for hash navigation
        const docName = url.split('/').pop().replace('.md', '');
        return `<a href="#doc-${docName}">${text}</a>`;
      }
      // Check if the URL is a relative path to a folder (ends with /)
      else if ((url.startsWith('./') || url.startsWith('../')) && (url.endsWith('/') || !url.includes('.'))) {
        // Convert to GitHub repository URL
        const repoUrl = this._convertToGitHubUrl(url);
        return `<a href="${repoUrl}" target="_blank" rel="noopener noreferrer">${text}</a>`;
      }
      // Regular external link
      return `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    });
    
    // Process images
    html = html.replace(this.patterns.image, (match, alt, src) => {
      return `<img src="${src}" alt="${alt}" />`;
    });
    
    // Process lists
    const processedLines = [];
    const lines = html.split('\n');
    
    let inOrderedList = false;
    let inUnorderedList = false;
    let listStartNumber = 1; // Default start number
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Check for unordered list items
      const unorderedMatch = line.match(/^\-\s+(.+)$/);
      if (unorderedMatch) {
        if (!inUnorderedList) {
          processedLines.push('<ul>');
          inUnorderedList = true;
        }
        processedLines.push(`<li>${unorderedMatch[1].trim()}</li>`);
        continue;
      } else if (inUnorderedList) {
        processedLines.push('</ul>');
        inUnorderedList = false;
      }
      
      // Check for ordered list items - capture the number
      const orderedMatch = line.match(/^(\d+)\.\s+(.+)$/);
      if (orderedMatch) {
        if (!inOrderedList) {
          // Get the starting number for the list
          listStartNumber = parseInt(orderedMatch[1], 10);
          processedLines.push(`<ol start="${listStartNumber}">`);
          inOrderedList = true;
        }
        processedLines.push(`<li>${orderedMatch[2].trim()}</li>`);
        continue;
      } else if (inOrderedList) {
        processedLines.push('</ol>');
        inOrderedList = false;
      }
      
      // If not a list item, add the line as is
      processedLines.push(line);
    }
    
    // Close any open lists
    if (inUnorderedList) {
      processedLines.push('</ul>');
    }
    if (inOrderedList) {
      processedLines.push('</ol>');
    }
    
    html = processedLines.join('');
    
    // Process horizontal rules
    html = html.replace(this.patterns.horizontalRule, '<hr>');
    
    // Restore code blocks with improved formatting
    codeBlocks.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      const languageClass = block.language ? `language-${block.language}` : '';
      const formattedCode = this._formatCodeBlock(block.code, block.language);
      
      // Add data-language attribute for styling
      const dataLanguage = block.language ? ` data-language="${block.language}"` : '';
      
      // Add scroll message and copy button
      const scrollMessage = '<div class="scroll-message">← scroll using keyboard →</div>';
      const copyButton = '<button class="copy-button" title="Copy to clipboard">Copy</button>';
      
      html = html.replace(
        placeholder,
        `<pre class="code-block with-line-numbers"${dataLanguage}>${copyButton}<div class="line-numbers"></div><code class="${languageClass}" data-code-content="${this._escapeHtml(block.code)}">${formattedCode}</code>${scrollMessage}</pre>`
      );
    });
    
    // Final cleanup
    html = this._postprocessHtml(html);
    
    return html;
  }
  
  /**
   * Convert a relative path to a GitHub repository URL
   * @param {string} url - The relative URL to convert
   * @returns {string} The GitHub repository URL
   */
  _convertToGitHubUrl(url) {
    // Remove leading ./ or ../
    let path = url.replace(/^\.\//, '').replace(/^\.\.\//, '');
    
    // Remove trailing slash if present
    path = path.replace(/\/$/, '');
    
    // Construct the GitHub URL
    return `https://github.com/ddttom/mlx-llm-tutorial/tree/main/${path}`;
  }
  
  /**
   * Process markdown tables
   * @param {string} html - The HTML to process
   * @returns {string} The processed HTML with tables
   */
  _processTables(html) {
    // Find table blocks
    const lines = html.split('\n');
    const processedLines = [];
    
    let inTable = false;
    let tableRows = [];
    let isHeaderRow = false;
    let alignments = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Check if this is a table row
      if (line.startsWith('|') && line.endsWith('|')) {
        // If not already in a table, start a new one
        if (!inTable) {
          inTable = true;
          tableRows = [];
          isHeaderRow = true; // First row is assumed to be header
        }
        
        // Check if this is a separator row (e.g., |---|---|)
        const separatorMatch = line.match(/^\|((?:[ \t]*:?-+:?[ \t]*\|)+)$/);
        if (separatorMatch) {
          // Parse column alignments from separator row
          const separators = separatorMatch[1].split('|').filter(Boolean);
          alignments = separators.map(sep => {
            if (sep.trim().startsWith(':') && sep.trim().endsWith(':')) return 'center';
            if (sep.trim().endsWith(':')) return 'right';
            return 'left';
          });
          continue; // Skip the separator row
        }
        
        // Process the table row
        const cells = line.slice(1, -1).split('|').map(cell => cell.trim());
        tableRows.push(cells);
        
        // If this is the first row and the next row is not a separator, it's not a header
        if (isHeaderRow && i + 1 < lines.length) {
          const nextLine = lines[i + 1].trim();
          if (!nextLine.match(/^\|((?:[ \t]*:?-+:?[ \t]*\|)+)$/)) {
            isHeaderRow = false;
          }
        }
        
        continue;
      } else if (inTable) {
        // End of table
        inTable = false;
        
        // Generate the HTML table
        const tableHtml = this._generateTableHtml(tableRows, isHeaderRow, alignments);
        processedLines.push(tableHtml);
      }
      
      // If not a table row, add the line as is
      if (!inTable) {
        processedLines.push(line);
      }
    }
    
    // If we're still in a table at the end, close it
    if (inTable) {
      const tableHtml = this._generateTableHtml(tableRows, isHeaderRow, alignments);
      processedLines.push(tableHtml);
    }
    
    return processedLines.join('\n');
  }
  
  /**
   * Generate HTML table from markdown table rows
   * @param {Array<Array<string>>} rows - The table rows
   * @param {boolean} hasHeader - Whether the table has a header row
   * @param {Array<string>} alignments - Column alignments (left, center, right)
   * @returns {string} The HTML table
   */
  _generateTableHtml(rows, hasHeader, alignments) {
    if (rows.length === 0) return '';
    
    let html = '<table><tbody>';
    
    // Add table header if present
    if (hasHeader && rows.length > 0) {
      const headerRow = rows[0];
      html += '<tr>';
      headerRow.forEach((cell, index) => {
        const align = alignments[index] ? ` style="text-align: ${alignments[index]}"` : '';
        html += `<td${align}>${cell}</td>`;
      });
      html += '</tr>';
      rows = rows.slice(1); // Remove header row from data rows
    }
    
    // Add table body
    if (rows.length > 0) {
      rows.forEach(row => {
        html += '<tr>';
        row.forEach((cell, index) => {
          const align = alignments[index] ? ` style="text-align: ${alignments[index]}"` : '';
          html += `<td${align}>${cell}</td>`;
        });
        html += '</tr>';
      });
    }
    
    html += '</tbody></table>';
    return html;
  }
  
  /**
   * Format code block for better display
   * @param {string} code - The code to format
   * @param {string} language - The language of the code
   * @returns {string} The formatted code
   */
  _formatCodeBlock(code, language) {
    // Split the code into lines for better formatting
    const lines = code.split('\n');
    
    // Join the lines back together with proper line breaks
    return lines.join('\n');
  }
  
  /**
   * Preprocess markdown to clean it up
   * @param {string} markdown - The markdown text to preprocess
   * @returns {string} The preprocessed markdown
   */
  _preprocessMarkdown(markdown) {
    return markdown
      .replace(/\r\n/g, '\n') // Normalize line endings
      .replace(/\n{3,}/g, '\n\n') // Replace 3+ consecutive newlines with just 2
      .replace(/```([a-z]*)\n\n+/g, '```$1\n') // Remove blank lines after code block start
      .replace(/\n\n+```/g, '\n```') // Remove blank lines before code block end
      .replace(/\n\n+(\s*[-*+])/g, '\n$1') // Remove blank lines before list items
      .trim(); // Remove leading/trailing whitespace
  }
  
  /**
   * Postprocess HTML to clean it up
   * @param {string} html - The HTML to postprocess
   * @returns {string} The postprocessed HTML
   */
  _postprocessHtml(html) {
    return html
      .replace(/<p>\s*<\/p>/g, '') // Remove empty paragraphs
      .replace(/>\s+</g, '><') // Remove whitespace between tags
      .replace(/\s{2,}/g, ' ') // Replace multiple spaces with a single space
      .replace(/<\/li><li>/g, '</li>\n<li>') // Add newlines between list items for readability
      .replace(/<\/h(\d)><h\d>/g, '</h$1>\n<h$1>') // Add newlines between headers for readability
      .replace(/<\/pre><pre/g, '</pre>\n<pre'); // Add newlines between code blocks for readability
  }
  
  /**
   * Escape HTML special characters
   * @param {string} html - The HTML to escape
   * @returns {string} The escaped HTML
   */
  _escapeHtml(html) {
    return html
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

/**
 * Markdown Renderer
 * Renders markdown content from a URL into an HTML element
 */
class MarkdownRenderer {
  constructor() {
    this.parser = new MarkdownParser();
  }
  
  /**
   * Render markdown from a URL into a target element
   * @param {string} url - The URL of the markdown file
   * @param {string|HTMLElement} target - The target element selector or element
   * @param {Object} options - Additional options
   */
  async renderFromUrl(url, target, options = {}) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch markdown: ${response.status} ${response.statusText}`);
      }
      
      const markdown = await response.text();
      this.renderMarkdown(markdown, target, options);
    } catch (error) {
      console.error('Error rendering markdown:', error);
      const targetElement = typeof target === 'string' ? document.querySelector(target) : target;
      if (targetElement) {
        targetElement.innerHTML = `<div class="markdown-error">Error loading markdown: ${error.message}</div>`;
      }
    }
  }
  
  /**
   * Render markdown text into a target element
   * @param {string} markdown - The markdown text
   * @param {string|HTMLElement} target - The target element selector or element
   * @param {Object} options - Additional options
   */
  renderMarkdown(markdown, target, options = {}) {
    const html = this.parser.parse(markdown);
    const targetElement = typeof target === 'string' ? document.querySelector(target) : target;
    
    if (targetElement) {
      targetElement.innerHTML = html;
      
      // Apply syntax highlighting if a highlighter is available
      if (options.highlight && typeof options.highlight === 'function') {
        const codeBlocks = targetElement.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
          options.highlight(block);
        });
      }
      
      // Add line numbers to code blocks
      this._addLineNumbersToCodeBlocks(targetElement);
      
      // Add copy to clipboard functionality
      this._addCopyToClipboardFunctionality(targetElement);
      
      // Process markdown links to use hash navigation
      this._processMarkdownLinks(targetElement);
      
      // Dispatch event when rendering is complete
      targetElement.dispatchEvent(new CustomEvent('markdown-rendered', {
        detail: { markdown, html }
      }));
    }
  }
  
  /**
   * Decode HTML entities back to their original characters
   * @param {string} html - The HTML with entities to decode
   * @returns {string} The decoded HTML
   */
  _decodeHtml(html) {
    const textarea = document.createElement('textarea');
    textarea.innerHTML = html;
    return textarea.value;
  }

  /**
   * Process markdown links to use hash navigation
   * @param {HTMLElement} container - The container element
   */
  _processMarkdownLinks(container) {
    // Process markdown file links
    const markdownLinks = container.querySelectorAll('a[href$=".md"]');
    markdownLinks.forEach(link => {
      const href = link.getAttribute('href');
      const docName = href.split('/').pop().replace('.md', '');
      link.setAttribute('href', `#doc-${docName}`);
      
      // Prevent default and handle navigation manually
      link.addEventListener('click', (e) => {
        e.preventDefault();
        window.location.hash = `doc-${docName}`;
      });
    });
    
    // Process relative folder links
    const folderLinks = container.querySelectorAll('a[href^="./"], a[href^="../"]');
    folderLinks.forEach(link => {
      const href = link.getAttribute('href');
      // Check if it's a folder link (ends with / or doesn't have a file extension)
      if (href.endsWith('/') || !href.includes('.')) {
        // Convert to GitHub repository URL
        const repoUrl = this._convertToGitHubUrl(href);
        link.setAttribute('href', repoUrl);
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }
    });
  }
  
  /**
   * Convert a relative path to a GitHub repository URL
   * @param {string} url - The relative URL to convert
   * @returns {string} The GitHub repository URL
   */
  _convertToGitHubUrl(url) {
    // Remove leading ./ or ../
    let path = url.replace(/^\.\//, '').replace(/^\.\.\//, '');
    
    // Remove trailing slash if present
    path = path.replace(/\/$/, '');
    
    // Construct the GitHub URL
    return `https://github.com/ddttom/mlx-llm-tutorial/tree/main/${path}`;
  }
  
  /**
   * Add line numbers to code blocks
   * @param {HTMLElement} container - The container element
   */
  _addLineNumbersToCodeBlocks(container) {
    const codeBlocks = container.querySelectorAll('pre code');
    codeBlocks.forEach(codeBlock => {
      // Get the code content
      const content = codeBlock.textContent;
      
      // Split the content into lines
      const lines = content.split('\n');
      
      // Get the line numbers wrapper
      const lineNumbersWrapper = codeBlock.parentElement.querySelector('.line-numbers');
      if (!lineNumbersWrapper) return;
      
      // Create the line numbers
      for (let i = 1; i <= lines.length; i++) {
        const lineNumber = document.createElement('span');
        lineNumber.className = 'line-number';
        lineNumber.textContent = i;
        lineNumbersWrapper.appendChild(lineNumber);
      }
    });
  }
  
  /**
   * Add copy to clipboard functionality to code blocks
   * @param {HTMLElement} container - The container element
   */
  _addCopyToClipboardFunctionality(container) {
    const copyButtons = container.querySelectorAll('.copy-button');
    copyButtons.forEach(button => {
      button.addEventListener('click', () => {
        // Get the code content
        const codeElement = button.parentElement.querySelector('code');
        const codeContent = codeElement.getAttribute('data-code-content');
        
        // Use a fallback method for copying text to clipboard
        this._copyTextToClipboard(codeContent, button);
      });
    });
  }
  
  /**
   * Copy text to clipboard using a fallback method
   * @param {string} text - The text to copy
   * @param {HTMLElement} button - The button element
   */
  _copyTextToClipboard(text, button) {
    try {
      // Decode HTML entities before copying
      const decodedText = this._decodeHtml(text);
      
      // Create a temporary textarea element
      const textarea = document.createElement('textarea');
      textarea.value = decodedText;
      
      // Make the textarea out of viewport
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      textarea.style.top = '0';
      
      // Append the textarea to the document
      document.body.appendChild(textarea);
      
      // Select the text
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);
      
      // Copy the text
      const successful = document.execCommand('copy');
      
      // Remove the textarea
      document.body.removeChild(textarea);
      
      // Show success or failure message
      if (successful) {
        button.textContent = 'Copied!';
        button.classList.add('copied');
      } else {
        button.textContent = 'Failed';
        button.classList.add('failed');
      }
      
      // Reset button text after 2 seconds
      setTimeout(() => {
        button.textContent = 'Copy';
        button.classList.remove('copied', 'failed');
      }, 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
      button.textContent = 'Failed';
      button.classList.add('failed');
      
      // Reset button text after 2 seconds
      setTimeout(() => {
        button.textContent = 'Copy';
        button.classList.remove('failed');
      }, 2000);
    }
  }
}

// Export the renderer as a global variable and as a module
window.MarkdownRenderer = new MarkdownRenderer();

// Export as ES module
export default new MarkdownRenderer();