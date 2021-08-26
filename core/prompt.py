import random
import itertools


def get_permuted_prompts(text_input, n, seed=None):
    
    def get_permutations(string, lists, n, seed=None):
        combos = list(itertools.product(*lists))
        texts = [string.format(*c) for c in combos]
        if seed is not None:
            random.seed(seed)
        random.shuffle(texts)
        return texts[:n]

    status = ['trending', 'popular', 'favorite', 'loved']
    media = ['photograph', 'photo', 'painting', 'vector drawing', 'illustration', 
             'rendering', 'sketch', 'pencil sketch', 'drawing', 'artwork', 'animation', 'picture']
    verbs = ['rendered', 'raytraced', 'designed', 'animated', 'created', 
             'drawn', 'illustrated', 'painted', 'sketched', 'photographed']
    adjectives = ['brilliant', 'photorealistic', 'hyper-realistic', 'detailed', 
                  'enhanced', 'exquisite', 'sharp', 'detailed', 'beautiful', 
                  'epic', 'gorgeous', 'amazing', 'breathtaking', 'incredible', 'stunning']
    descriptions = ['4k resolution', 'HDR', 'desktop wallpaper', 'remastered', 
                    'ultra high-resolution', 'mixed media', 'raytracing', 
                    'DSLR', 'VFX', 'CGI']
    engines = ['Blender', 'Unreal Engine', 'Unity', 'Cinema4D', 
               'Adobe After Effects', 'Adobe Illustrator', 'Photoshop', 'Maya', 
               'Processing', 'openFrameworks', 'Flash']
    platforms = ['National Geographic', 'Associated press', 'Pixar movie', 
                 'Artstation', 'DeviantArt', 'Pinterest', 'Instagram', '/r/art',
                 '/r/EarthPorn', 'Behance', 'Dayflash', 'Dribble']

    permutations = get_permutations(
        '{} {} of {} {} in {}', 
        [adjectives, media, [text_input], verbs, engines], n, seed)
    permutations += get_permutations(
        '{} {} of {} {}', 
        [adjectives, media, [text_input], descriptions], n, seed)
    permutations += get_permutations(
        '{} of {} {} on {}', 
        [media, [text_input], status, platforms], n, seed)

    text_inputs = [{'text': p, 'weight': random.random()} for p in permutations]
    return text_inputs
