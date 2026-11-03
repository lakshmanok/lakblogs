/* Copy only the biography paragraphs, preserving paragraph breaks. */
'use strict';
document.querySelectorAll('button[data-copy]').forEach((button) => {
  const source = document.getElementById(button.dataset.copy);
  const status = document.getElementById(button.getAttribute('aria-describedby'));
  if (!source || !status) return;
  button.hidden = false;
  button.addEventListener('click', async () => {
    const text = Array.from(source.querySelectorAll('p'), (p) => p.textContent.trim()).join('\n\n');
    button.disabled = true;
    status.textContent = '';
    try {
      await navigator.clipboard.writeText(text);
      status.textContent = 'Copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      if (selection) {
        const range = document.createRange();
        range.selectNodeContents(source);
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Text selected. Use your device’s Copy command.';
      } else {
        status.textContent = 'Please select the biography text and copy it manually.';
      }
    } finally {
      button.disabled = false;
    }
  });
});
