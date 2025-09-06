import os
import sys

import __web.deepzoom_server as deepzoom
import __web.start_web as web

import config


def _check_seadragon_assets():
    """Ensure required OpenSeadragon assets are present before starting the server.

    Files checked (relative to project root):
      - __web/static/openseadragon.js
      - __web/static/openseadragon-scalebar.js
    """
    missing = []
    required = [
        os.path.join('__web', 'static', 'jquery.js'),
        os.path.join('__web', 'static', 'openseadragon.js'),
        os.path.join('__web', 'static', 'openseadragon-scalebar.js'),
    ]
    for path in required:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print('ERROR: Required OpenSeadragon files are missing:', file=sys.stderr)
        for m in missing:
            print(f' - {m}', file=sys.stderr)
        print('\nPlease add these files, otherwise the web preview will not work.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    _check_seadragon_assets()
    web.run(config.HDD_SLIDES)
