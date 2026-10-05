param(
    [string]$ToolchainRoot = '',
    [string]$BuildDirectory = ''
)
$ErrorActionPreference = 'Stop'
$taskRepository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (!$ToolchainRoot) { $ToolchainRoot = Join-Path $taskRepository 'artifacts\yolo1050\toolchain' }
if (!$BuildDirectory) { $BuildDirectory = Join-Path $PSScriptRoot 'build\release-11.8' }
$ToolchainRoot = [IO.Path]::GetFullPath($ToolchainRoot)
$taskCuda = Join-Path $ToolchainRoot 'cuda-11.8'
$taskMsvc = Join-Path $ToolchainRoot 'msvc-2019\VC\Tools\MSVC\14.29.30133'
$taskCompiler = Join-Path $taskMsvc 'bin\Hostx64\x64\cl.exe'
$taskTrt = Join-Path $ToolchainRoot 'sdk\TensorRT-8.6.1.6'
$taskOpenCV = Join-Path $ToolchainRoot 'opencv-sdk\opencv\build'
$taskWindowsSdk = 'C:\Program Files (x86)\Windows Kits\10'
$taskWindowsVersion = (Get-ChildItem -LiteralPath (Join-Path $taskWindowsSdk 'Lib') -Directory |
    Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1).Name
foreach ($taskRequired in @($taskCompiler, "$taskCuda\bin\nvcc.exe", "$taskTrt\include\NvInfer.h",
                            "$taskOpenCV\OpenCVConfig.cmake", "$taskWindowsSdk\Lib\$taskWindowsVersion\um\x64")) {
    if (!(Test-Path -LiteralPath $taskRequired)) { throw "Build dependency missing: $taskRequired" }
}
$taskSavedEnvironment = @{}
foreach ($taskName in @('PATH','INCLUDE','LIB','CUDA_PATH','CUDACXX','CUDAHOSTCXX','NVCC_PREPEND_FLAGS')) {
    $taskSavedEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
}
try {
    $env:PATH = "$taskMsvc\bin\Hostx64\x64;$taskWindowsSdk\bin\$taskWindowsVersion\x64;$taskCuda\bin;" + $env:PATH
    $env:INCLUDE = "$taskMsvc\include;$taskWindowsSdk\Include\$taskWindowsVersion\ucrt;$taskWindowsSdk\Include\$taskWindowsVersion\shared;$taskWindowsSdk\Include\$taskWindowsVersion\um;$taskWindowsSdk\Include\$taskWindowsVersion\winrt"
    $env:LIB = "$taskMsvc\lib\x64;$taskWindowsSdk\Lib\$taskWindowsVersion\ucrt\x64;$taskWindowsSdk\Lib\$taskWindowsVersion\um\x64"
    $env:CUDA_PATH = $taskCuda
    $env:CUDACXX = "$taskCuda\bin\nvcc.exe"
    $env:CUDAHOSTCXX = $taskCompiler
    # The private compiler's SDK environment is already configured above.
    $env:NVCC_PREPEND_FLAGS = '--use-local-env'
    & cmake -S $PSScriptRoot -B $BuildDirectory -G Ninja '-DCMAKE_BUILD_TYPE=Release' `
        "-DCMAKE_CXX_COMPILER=$taskCompiler" "-DCMAKE_CUDA_HOST_COMPILER=$taskCompiler" `
        "-DCMAKE_CUDA_COMPILER=$taskCuda\bin\nvcc.exe" "-DCUDAToolkit_ROOT=$taskCuda" `
        "-DTENSORRT_ROOT=$taskTrt" "-DOpenCV_DIR=$taskOpenCV" '-DJSON_BuildTests=OFF'
    if ($LASTEXITCODE -ne 0) { throw "Native build configuration failed: $LASTEXITCODE" }
    & cmake --build $BuildDirectory --config Release --parallel 4
    if ($LASTEXITCODE -ne 0) { throw "Native compilation failed: $LASTEXITCODE" }
    # cuDNN's Windows runtime imports zlibwapi.dll. Build only that library,
    # never zlib's default target (which also builds example/test programs).
    $taskZlibSource = Join-Path $ToolchainRoot 'zlib-src\zlib-1.3.2'
    $taskZlibRuntime = Join-Path $ToolchainRoot 'zlib-runtime'
    if (!(Test-Path -LiteralPath "$taskZlibSource\win32\Makefile.msc")) {
        throw "Runtime dependency source missing: $taskZlibSource"
    }
    Push-Location -LiteralPath $taskZlibSource
    try {
        $taskZlibObjects = @('adler32','compress','crc32','deflate','gzclose','gzlib','gzread',
            'gzwrite','infback','inflate','inftrees','inffast','trees','uncompr','zutil') |
            ForEach-Object { "$_.obj" }
        & "$taskMsvc\bin\Hostx64\x64\nmake.exe" -nologo -f win32/Makefile.msc `
            'LOC=-DZLIB_WINAPI' @taskZlibObjects zlib1.res
        if ($LASTEXITCODE -ne 0) { throw "zlib runtime compilation failed: $LASTEXITCODE" }
        # cuDNN imports zlib by ordinal. Renaming zlib1.dll would have the wrong
        # entry points; retain the historic zlibwapi ordinals in our definition.
        & "$taskMsvc\bin\Hostx64\x64\link.exe" /NOLOGO /DLL /MACHINE:X64 /DYNAMICBASE /NXCOMPAT `
            /OPT:REF "/DEF:$PSScriptRoot\zlibwapi.def" /IMPLIB:zlibwapi.lib /OUT:zlibwapi.dll `
            @taskZlibObjects zlib1.res
        if ($LASTEXITCODE -ne 0) { throw "zlib runtime linking failed: $LASTEXITCODE" }
        [IO.Directory]::CreateDirectory($taskZlibRuntime) | Out-Null
        Copy-Item -LiteralPath 'zlibwapi.dll' -Destination $taskZlibRuntime
        Copy-Item -LiteralPath 'LICENSE' -Destination $taskZlibRuntime
    } finally { Pop-Location }
    Write-Output "Built $BuildDirectory\yolo1050.exe. No runtime or tests were started."
} finally {
    foreach ($taskName in $taskSavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($taskName, $taskSavedEnvironment[$taskName], 'Process')
    }
}
