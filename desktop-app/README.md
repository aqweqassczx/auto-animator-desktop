# Auto Animator Desktop (Tauri + React)

## Что получает конечный пользователь

Один установщик `.msi`:
- без ручной установки `pip`-пакетов,
- без ручного запуска Python-скриптов.

Python-часть упаковывается в `pipeline_runner.exe` и кладется внутрь инсталлятора как ресурс.

## Локальный запуск для разработки

1. Установить Node.js 20+, Rust stable, Python 3.11+.
2. В `desktop-app`:
   - `npm install`
   - `npm run tauri:dev`

## Сборка инсталлятора (локально)

1. Собрать встроенный раннер пайплайна:
   - `powershell -ExecutionPolicy Bypass -File .\tools\build_pipeline_runner.ps1`
2. Собрать Tauri-приложение:
   - `npm run tauri:build`
3. Готовый установщик:
   - `src-tauri/target/release/bundle/msi/`

## GitHub репозиторий и автообновления

Репозиторий: [aqweqassczx/auto-animator-desktop](https://github.com/aqweqassczx/auto-animator-desktop)

### Один раз настроить

1. Сгенерировать ключ подписи обновлений:
   - `tauri signer generate -w ~/.tauri/auto-animator.key`
2. Публичный ключ вставить в `src-tauri/tauri.conf.json` в `tauri.updater.pubkey`.
3. Приватный ключ добавить в GitHub Secrets:
   - `TAURI_PRIVATE_KEY`
   - `TAURI_KEY_PASSWORD` (если задан при генерации).

### Выпуск новой версии

1. Поднять версию в `src-tauri/tauri.conf.json` -> `package.version`.
2. Запустить workflow `desktop-release` в GitHub Actions.
3. Взять артефакты `.msi` (+ updater файлы) и прикрепить к GitHub Release.
4. После этого приложение у пользователей покажет кнопку обновления.
