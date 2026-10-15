# Saga's launch-or-focus: same script as Aiva's, different console.
# Shortcut target:
#   C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Users\Extre\Desktop\Work\UnseenMedia\Aiva_2.0\launch_or_focus_saga.ps1"
# (or point a shortcut straight at launch_or_focus.ps1 with `-Character saga`).
& (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "launch_or_focus.ps1") -Character saga
