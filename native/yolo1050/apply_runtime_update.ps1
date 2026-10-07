param([string]$InstallDirectory = '', [string]$RuntimeZip = '')
$ErrorActionPreference = 'Stop'

function Restore-RuntimeFiles {
    param([string]$ArchivePath, [string]$RuntimeDirectory, [string[]]$Names, $FileRecords)
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    $taskArchive = $null
    $taskStaged = @{}
    try {
        $taskArchive = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $ArchivePath).Path)
        $taskEntries = @{}
        foreach ($taskEntry in $taskArchive.Entries) {
            $taskEntryName = $taskEntry.FullName.Replace('\', '/')
            if ($taskEntryName -match '(?:^|/)runtime/(?<dll>[\w.-]+\.dll)$') {
                $taskName = $Matches.dll
                if ($Names -contains $taskName) {
                    if ($taskEntries.ContainsKey($taskName)) {
                        throw "The ZIP contains duplicate runtime copies of $taskName. Select the original full package ZIP."
                    }
                    $taskEntries[$taskName] = $taskEntry
                }
            }
        }
        foreach ($taskName in $Names) {
            $taskExpected = $FileRecords.PSObject.Properties[$taskName].Value
            if (!$taskEntries.ContainsKey($taskName) -or
                $taskEntries[$taskName].Length -ne $taskExpected.bytes) {
                throw "The ZIP does not contain the expected runtime/$taskName. Select the ORIGINAL friend-1080p-ready.zip, not the small update ZIP."
            }
        }
        [IO.Directory]::CreateDirectory($RuntimeDirectory) | Out-Null
        $taskToken = [Guid]::NewGuid().ToString('N')
        $taskIndex = 0
        foreach ($taskName in $Names) {
            $taskIndex++
            Write-Host "Restoring [$taskIndex/$($Names.Count)]: $taskName"
            $taskTemporary = Join-Path $RuntimeDirectory ".restore-$taskToken-$taskName.tmp"
            $taskStaged[$taskName] = $taskTemporary
            $taskInput = $null
            $taskOutput = $null
            try {
                $taskInput = $taskEntries[$taskName].Open()
                $taskOutput = [IO.File]::Open($taskTemporary, [IO.FileMode]::CreateNew,
                    [IO.FileAccess]::Write, [IO.FileShare]::None)
                $taskInput.CopyTo($taskOutput)
            } finally {
                if ($taskOutput) { $taskOutput.Dispose() }
                if ($taskInput) { $taskInput.Dispose() }
            }
            $taskExpected = $FileRecords.PSObject.Properties[$taskName].Value
            if ((Get-FileHash -LiteralPath $taskTemporary -Algorithm SHA256).Hash -ne $taskExpected.sha256) {
                throw "The restored $taskName does not match the original package hash. Download the original ZIP again."
            }
        }
        # Install only after every extracted file has matched the original package.
        foreach ($taskName in $Names) {
            $taskDestination = Join-Path $RuntimeDirectory $taskName
            if (Test-Path -LiteralPath $taskDestination) {
                throw "The destination changed during recovery: $taskDestination. Close other installers and retry."
            }
        }
        foreach ($taskName in $Names) {
            Move-Item -LiteralPath $taskStaged[$taskName] -Destination (Join-Path $RuntimeDirectory $taskName)
        }
    } finally {
        if ($taskArchive) { $taskArchive.Dispose() }
        foreach ($taskTemporary in $taskStaged.Values) {
            if (Test-Path -LiteralPath $taskTemporary -PathType Leaf) {
                Remove-Item -LiteralPath $taskTemporary -Force
            }
        }
    }
}

try {
    $taskRecord = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime-update.json') -Raw | ConvertFrom-Json
    $taskSource = Join-Path $PSScriptRoot 'runtime\yolo1050.exe'
    $taskExpectedHash = $taskRecord.files.'runtime/yolo1050.exe'.sha256
    if ((Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash -ne $taskExpectedHash) {
        throw 'The update executable does not match its manifest. Extract the entire update ZIP again.'
    }
    foreach ($taskName in $taskRecord.install_root_files) {
        if ($taskName -notin @('4-view-detections-fp32.cmd', '4-view-detections-int8.cmd')) {
            throw 'Invalid root file in update manifest.'
        }
        $taskFileRecord = $taskRecord.files.PSObject.Properties[$taskName].Value
        if (!$taskFileRecord -or (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $taskName) -Algorithm SHA256).Hash -ne $taskFileRecord.sha256) {
            throw "The update launcher does not match its manifest: $taskName. Extract the entire update ZIP again."
        }
    }
    if (!$InstallDirectory) {
        if ((Test-Path -LiteralPath (Join-Path $PSScriptRoot 'settings.json')) -and
            (Test-Path -LiteralPath (Join-Path $PSScriptRoot '3-start-int8.cmd'))) {
            $InstallDirectory = $PSScriptRoot
        } else {
            Add-Type -AssemblyName System.Windows.Forms
            $taskDialog = New-Object System.Windows.Forms.FolderBrowserDialog
            try {
                $taskDialog.Description = 'Select the ORIGINAL full package folder containing 3-start-int8.cmd and settings.json.'
                $taskDialog.ShowNewFolderButton = $false
                if ($taskDialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
                    Write-Host 'Update cancelled.'
                    exit 1
                }
                $InstallDirectory = $taskDialog.SelectedPath
            } finally { $taskDialog.Dispose() }
        }
    }
    $taskInstall = (Resolve-Path -LiteralPath $InstallDirectory).Path
    foreach ($taskName in @('3-start-int8.cmd','settings.json')) {
        if (!(Test-Path -LiteralPath (Join-Path $taskInstall $taskName) -PathType Leaf)) {
            throw "Select the original full package folder. Missing: $taskName"
        }
    }
    $taskRuntime = Join-Path $taskInstall 'runtime'
    Write-Host "Selected installation: $taskInstall"
    Write-Host "Required DLL location: $taskRuntime"
    if (Get-Process -Name yolo1050 -ErrorAction SilentlyContinue) {
        throw 'Close the detector before applying the update.'
    }
    $taskMissing = @()
    foreach ($taskName in $taskRecord.required_runtime_dlls) {
        if ($taskName -notmatch '^[\w.-]+\.dll$') { throw 'Invalid dependency name in update manifest.' }
        $taskExpected = $taskRecord.required_runtime_files.PSObject.Properties[$taskName].Value
        if (!$taskExpected -or $taskExpected.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or $taskExpected.bytes -le 0) {
            throw 'The update manifest is missing runtime recovery hashes. Extract the entire latest update ZIP again.'
        }
        if (!(Test-Path -LiteralPath (Join-Path $taskRuntime $taskName) -PathType Leaf)) { $taskMissing += $taskName }
    }
    if ($taskMissing.Count) {
        Write-Host "The commands and settings were found. Missing runtime DLLs: $($taskMissing.Count)/$($taskRecord.required_runtime_dlls.Count)." -ForegroundColor Yellow
        Write-Host 'Select the ORIGINAL 2.1 GB friend-1080p-ready ZIP to restore the missing DLLs automatically.'
        Write-Host 'Recovery reads only runtime DLLs; existing engines, settings, model, and calibration are retained.'
        if (!$RuntimeZip) {
            Add-Type -AssemblyName System.Windows.Forms
            $taskZipDialog = New-Object System.Windows.Forms.OpenFileDialog
            try {
                $taskZipDialog.Title = 'Select the ORIGINAL friend-1080p-ready.zip to restore missing DLLs'
                $taskZipDialog.Filter = 'ZIP archives (*.zip)|*.zip'
                $taskZipDialog.CheckFileExists = $true
                $taskZipDialog.Multiselect = $false
                $taskZipDialog.InitialDirectory = Split-Path -Parent $taskInstall
                if ($taskZipDialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
                    throw "Recovery cancelled. Commands/settings are present, but missing DLLs belong in: $taskRuntime"
                }
                $RuntimeZip = $taskZipDialog.FileName
            } finally { $taskZipDialog.Dispose() }
        }
        Restore-RuntimeFiles -ArchivePath $RuntimeZip -RuntimeDirectory $taskRuntime `
            -Names $taskMissing -FileRecords $taskRecord.required_runtime_files
        Write-Host "Restored $($taskMissing.Count) runtime DLLs."
    }
    if (Get-Process -Name yolo1050 -ErrorAction SilentlyContinue) {
        throw 'Close the detector before applying the executable update.'
    }
    $taskTarget = Join-Path $taskRuntime 'yolo1050.exe'
    if ([IO.Path]::GetFullPath($taskSource) -ne [IO.Path]::GetFullPath($taskTarget)) {
        $taskBackup = "$taskTarget.before-capture-update"
        if ((Test-Path -LiteralPath $taskTarget) -and !(Test-Path -LiteralPath $taskBackup)) {
            Copy-Item -LiteralPath $taskTarget -Destination $taskBackup
        }
        Copy-Item -LiteralPath $taskSource -Destination $taskTarget -Force
    }
    $taskNotes = Join-Path $PSScriptRoot 'CAPTURE-FIX.txt'
    $taskNotesTarget = Join-Path $taskInstall 'CAPTURE-FIX.txt'
    if ([IO.Path]::GetFullPath($taskNotes) -ne [IO.Path]::GetFullPath($taskNotesTarget)) {
        Copy-Item -LiteralPath $taskNotes -Destination $taskNotesTarget -Force
    }
    foreach ($taskName in $taskRecord.install_root_files) {
        $taskLauncherSource = Join-Path $PSScriptRoot $taskName
        $taskLauncherTarget = Join-Path $taskInstall $taskName
        if ([IO.Path]::GetFullPath($taskLauncherSource) -ne [IO.Path]::GetFullPath($taskLauncherTarget)) {
            if (Test-Path -LiteralPath $taskLauncherTarget) {
                $taskLauncherBackup = "$taskLauncherTarget.before-runtime-update"
                if (!(Test-Path -LiteralPath $taskLauncherBackup)) {
                    Copy-Item -LiteralPath $taskLauncherTarget -Destination $taskLauncherBackup
                }
            }
            Copy-Item -LiteralPath $taskLauncherSource -Destination $taskLauncherTarget -Force
        }
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'runtime-update.json') `
        -Destination (Join-Path $taskInstall 'runtime-update-installed.json') -Force
    Write-Host "Updated: $taskTarget"
    Write-Host 'Run 3-start-int8.cmd from the ORIGINAL package folder. Existing engines and calibration are retained.'
    Write-Host 'Detection views: 4-view-detections-fp32.cmd / 4-view-detections-int8.cmd (boxes and scores; no aiming or shooting).'
    exit 0
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
