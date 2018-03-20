from argparse import ArgumentParser
from utils import config, read_config_file

parser = ArgumentParser()
parser.add_argument('--cuda', help='Is cuda?', action='store_true')
parser.add_argument('--model', help='Model file path', required=True)
parser.add_argument('--config', help='Path to config.json', default='config.json')
parser.add_argument('--test', help='Use fake word2vec', action='store_true')
parser.add_argument('--port', type=int, help='List to this port')

args = parser.parse_args()

read_config_file(args.config)
config.update(args.__dict__)

from webapp import app
app.run(debug=True, port=args.port)

