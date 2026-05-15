# Compatibility wrapper. The old start_all.ps1 was Qwen-era and also tried to
# launch emulator/app UI. Echo is Gemma-first now, so delegate to the current
# service launcher.

param(
    [switch]$SkipGemma,
    [switch]$SkipLiveKit,
    [switch]$SkipEcho,
    [switch]$WithVoice,
    [switch]$SkipVoiceAgent,
    [switch]$KeepQwen,
    [int]$GemmaTimeoutSeconds = 600,
    [int]$EchoTimeoutSeconds = 90
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Root "start_echo_services.ps1") `
    -SkipGemma:$SkipGemma `
    -SkipLiveKit:$SkipLiveKit `
    -SkipEcho:$SkipEcho `
    -WithVoice:$WithVoice `
    -SkipVoiceAgent:$SkipVoiceAgent `
    -KeepQwen:$KeepQwen `
    -GemmaTimeoutSeconds $GemmaTimeoutSeconds `
    -EchoTimeoutSeconds $EchoTimeoutSeconds
