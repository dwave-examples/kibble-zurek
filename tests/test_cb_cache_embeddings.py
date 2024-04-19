# Copyright 2024 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import pytest
from io import StringIO

from contextvars import copy_context, ContextVar
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash import no_update

from app import cache_embeddings

qpu_name = ContextVar('qpu_name')
embeddings_found = ContextVar('embeddings_found')
embeddings_cached = ContextVar('embeddings_cached')

embedding_filenames = [
    'emb_Advantage_system4.1.json',
    'emb_Advantage_system5.4.json',
    'emb_Advantage2_prototype2.3.json', ]

json_embeddings_file = '{ \
    "3": {"1": [11], "0": [10], "2": [12]}, \
    "5": {"1": [11], "0": [10], "2": [12], "3": [13], "4": [14]} }'

edges_5 = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 10)]
edges_3_5 = [(10, 11), (10, 12), (11, 12), (12, 13), (13, 14), (14, 10)]
 
class mock_qpu_edges():
    def __init__(self, edges):
        self.edges = edges

class mock_qpu(object):
    def __init__(self):
        self.edges_per_qpu = {
            'Advantage_system4.1': edges_5,
            'Advantage2_prototype2.55': edges_3_5, }
    
    def __getitem__(self, indx):
        return mock_qpu_edges(self.edges_per_qpu[indx])
    
parametrize_vals = [
    ('Advantage_system4.1', 
     embedding_filenames,
     json_embeddings_file,), 
    ('Advantage2_prototype2.55',
     embedding_filenames,
     json_embeddings_file,),
    ('Advantage88_prototype7.3',
     embedding_filenames,
     json_embeddings_file,), ]

@pytest.mark.parametrize(['qpu_name_val', 'embeddings', 'json_emb_file',],
parametrize_vals)
def test_cache_embeddings_qpu_selection(mocker, qpu_name_val, embeddings, json_emb_file,):
    """Test the caching of embeddings: triggered by QPU selection."""

    mocker.patch('app.os.listdir', return_value=embeddings)
    mocker.patch('builtins.open', return_value=StringIO(json_emb_file))
    mocker.patch('app.qpus', new=mock_qpu())

    def run_callback():
        context_value.set(AttributeDict(**
            {'triggered_inputs': [{'prop_id': 'qpu_selection.value'},]}))

        return cache_embeddings(qpu_name.get(), embeddings_found.get(), embeddings_cached.get())

    qpu_name.set(qpu_name_val)
    embeddings_found.set('dummy')
    embeddings_cached.set('dummy')

    ctx = copy_context()
    output = ctx.run(run_callback)

    if qpu_name_val == 'Advantage_system4.1':
        assert output[1] == [5]
    
    if qpu_name_val == 'Advantage2_prototype2.55':
        assert output[1] == [3, 5]

    if qpu_name_val == 'Advantage88_prototype7.3':
        assert output == ({}, [])

parametrize_vals = [
    ('{"22": {"1": [11], "0": [10], "2": [12]}}', 
     json_embeddings_file,), 
    ('needed',
     json_embeddings_file,),
    ('not found',
     json_embeddings_file,), ]

@pytest.mark.parametrize(['embeddings_found_val', 'embeddings_cached_val'],
parametrize_vals)
def test_cache_embeddings_found_embedding(embeddings_found_val, embeddings_cached_val):
    """Test the caching of embeddings: triggered by found embedding."""

    def run_callback():
        context_value.set(AttributeDict(**
            {'triggered_inputs': [{'prop_id': 'embeddings_found.data'},]}))

        return cache_embeddings(qpu_name.get(), embeddings_found.get(), embeddings_cached.get())

    qpu_name.set("dummy")
    embeddings_found.set(embeddings_found_val)
    embeddings_cached.set(embeddings_cached_val)

    ctx = copy_context()
    output = ctx.run(run_callback)

    if not isinstance(embeddings_found_val, dict):
        assert output == (no_update, no_update)
    else:
        assert 22 in output[1]
