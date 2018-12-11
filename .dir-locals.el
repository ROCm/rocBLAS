;; This causes emacs to abide by formatting rules: No tab characters, except
;; in Makefiles; BSD-style indendation, with a 4-character indentation offset
;; (8 in assembly code and Makefiles); no whitespace at the end of lines; and
;; a newline at the end of file.

(
 (nil . (
  (require-final-newline . t)
  (indent-tabs-mode . nil)
  (eval add-hook 'before-save-hook 'delete-trailing-whitespace)
  (eval add-hook 'before-save-hook (lambda () (untabify (point-min) (point-max))))
 ) )
 (prog-mode . (
  (tab-width . 4)
  (c-basic-offset . 4)
  (c-file-style . "bsd")
 ) )
 (asm-mode . (
  (tab-width . 8)
 ) )
 (makefile-mode . (
  (indent-tabs-mode . t)
  (tab-width . 8)
  (eval remove-hook `before-save-hook (lambda () (untabify (point-min) (point-max))))
 ) )
 (fundamental-mode . (
  (eval require 'yaml-mode)
  (eval set-auto-mode t)
  ) )
)
